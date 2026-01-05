"""Helpers for parsing and formatting tool calls."""

import json
import logging
from typing import Any, Dict, List

_logger = logging.getLogger(__name__)


def _normalize_call(parsed_obj: Dict[str, Any] | None, tool_call_obj) -> Dict[str, Any]:
    parsed_obj = parsed_obj or {}
    sim_type = parsed_obj.get("simulation_type")
    analysis = parsed_obj.get("analysis_type")
    sim_tool = parsed_obj.get("simulation_tool")

    if not sim_tool:
        sim_tool = getattr(tool_call_obj.function, "name", None) or getattr(tool_call_obj, "name", None)

    if str(sim_tool).lower() == "universal_circuit_tool":
        sim_tool = "run_ngspice"

    return {
        "simulation_type": str(sim_type).lower() if sim_type else sim_type,
        "analysis_type": str(analysis).lower() if analysis else analysis,
        "simulation_tool": str(sim_tool).lower() if sim_tool else sim_tool,
        "raw_args": parsed_obj,
    }


def _parse_arguments(arguments: Any) -> List[Dict[str, Any]]:
    """Safely parse tool call arguments into a list of dicts, without eval."""
    if isinstance(arguments, dict):
        return [arguments]
    if isinstance(arguments, list):
        return [item for item in arguments if isinstance(item, dict)]
    if not isinstance(arguments, str):
        return []

    arg_str = arguments.strip()
    if not arg_str:
        return []

    # First try direct JSON
    try:
        parsed = json.loads(arg_str)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception:
        pass

    # Next, try concatenated JSON objects: {}{}
    try:
        combined = "[" + arg_str.replace("}{", "},{") + "]"
        parsed_list = json.loads(combined)
        if isinstance(parsed_list, list):
            return [item for item in parsed_list if isinstance(item, dict)]
    except Exception:
        pass

    # Give up and return empty
    return []


def extract_tool_data(tool):
    """Extract tool call data from all tool calls in the tool object, safely."""
    message = tool.choices[0].message
    tool_data_list = []

    for tool_call in getattr(message, "tool_calls", []) or []:
        arguments = getattr(tool_call.function, "arguments", None)
        parsed_items = _parse_arguments(arguments)
        if not parsed_items:
            if isinstance(arguments, str) and not arguments.strip():
                continue  # silently ignore empty argument blobs
            tool_data_list.append(_normalize_call({}, tool_call))
            continue
        for parsed in parsed_items:
            tool_data_list.append(_normalize_call(parsed, tool_call))

    return tool_data_list


_ALLOWED_TOOL_NAMES = {
    "dc_simulation",
    "ac_simulation",
    "transient_simulation",
    "ac_gain",
    "dc_gain",
    "output_swing",
    "offset",
    "icmr",
    "tran_gain",
    "bandwidth",
    "unity_bandwidth",
    "phase_margin",
    "cmrr_tran",
    "power",
    "thd_input_range",
}

_ANALYSIS_TOOLS = {
    "ac_gain",
    "dc_gain",
    "output_swing",
    "offset",
    "icmr",
    "tran_gain",
    "bandwidth",
    "unity_bandwidth",
    "phase_margin",
    "cmrr_tran",
    "power",
    "thd_input_range",
}

# Mapping of analysis tools to the simulation that produces their data.
_REQUIRED_SIM_FOR_ANALYSIS = {
    "ac_gain": "ac_simulation",
    "bandwidth": "ac_simulation",
    "unity_bandwidth": "ac_simulation",
    "phase_margin": "ac_simulation",
    "dc_gain": "dc_simulation",
    "output_swing": "dc_simulation",
    "offset": "dc_simulation",
    "icmr": "dc_simulation",
    "tran_gain": "transient_simulation",
    "power": "transient_simulation",
    "thd_input_range": "transient_simulation",
    # cmrr_tran runs its own ngspice sweep, so no prerequisite simulation here.
}

_SIMULATION_TOOL_NAMES = [
    "dc_simulation",
    "ac_simulation",
    "transient_simulation",
]


def normalize_tool_chain(tool_chain: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool_calls: drop run_ngspice, dedupe sims, and inject required sims for analyses.

    - Removes any run_ngspice entries (ngspice is executed inside simulation_*).
    - Deduplicates simulation steps while preserving first occurrence order.
    - Ensures required simulation steps exist for analysis tools (inserting missing ones at the front).
    """
    if not isinstance(tool_chain, dict) or "tool_calls" not in tool_chain:
        return tool_chain

    calls = tool_chain.get("tool_calls") or []
    normalized = []
    seen_simulations = set()

    for call in calls:
        if not isinstance(call, dict):
            normalized.append(call)
            continue
        name_raw = call.get("name")
        name = str(name_raw).lower() if name_raw is not None else None
        if name == "run_ngspice":
            _logger.warning("Dropping run_ngspice tool call; simulation_* already runs ngspice.")
            continue

        if name in _SIMULATION_TOOL_NAMES:
            if name in seen_simulations:
                _logger.debug("Dropping duplicate simulation step: %s", name)
                continue
            seen_simulations.add(name)
            normalized.append({"name": name})
            continue

        # Keep analysis or other allowed entries; ensure name is lowercased
        normalized.append({"name": name} | {k: v for k, v in call.items() if k != "name"})

    # Ensure required simulations exist for analyses
    present_names = [c.get("name") for c in normalized if isinstance(c, dict)]
    required = []
    for call in normalized:
        if not isinstance(call, dict):
            continue
        sim_needed = _REQUIRED_SIM_FOR_ANALYSIS.get(call.get("name"))
        if sim_needed and sim_needed not in present_names and sim_needed not in required:
            required.append(sim_needed)

    if required:
        # Insert missing simulations at the front in deterministic order
        ordered_required = [sim for sim in _SIMULATION_TOOL_NAMES if sim in required]
        normalized = [{"name": sim} for sim in ordered_required] + normalized

    return {"tool_calls": normalized}


def validate_tool_chain(tool_chain: Dict[str, Any]) -> None:
    """Validate tool chain names and ordering before execution.

    Rules:
    - All tool names must be in the allowed list.
    - Analysis tools must appear after their prerequisite simulation step.
    - tool_calls must be a list of dicts with a 'name' field.
    """
    if not isinstance(tool_chain, dict) or "tool_calls" not in tool_chain:
        raise ValueError("tool_chain must be a dict with key 'tool_calls'")
    calls = tool_chain.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("tool_chain['tool_calls'] must be a list")

    seen_simulations = set()
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"tool call at index {idx} is not a dict")
        raw_name = call.get("name")
        name = str(raw_name).lower() if raw_name is not None else None
        if name not in _ALLOWED_TOOL_NAMES:
            raise ValueError(f"Unsupported tool '{raw_name}' at index {idx}")
        if name in {"dc_simulation", "ac_simulation", "transient_simulation"}:
            seen_simulations.add(name)
        required_sim = _REQUIRED_SIM_FOR_ANALYSIS.get(name)
        if required_sim and required_sim not in seen_simulations:
            raise ValueError(f"Analysis tool '{name}' requires '{required_sim}' earlier in the tool chain")



def _infer_simulation_type_from_analysis(analysis_type):
    """Infer a simulation type from an analysis_type string or list when simulation_type is missing.

    Rules (kept small and conservative):
    - AC-related analyses => 'ac'
    - transient analyses => 'transient'
    - DC-related analyses => 'dc'
    - If ambiguous or missing, default to 'dc' (safe choice for operating point checks).
    """
    if not analysis_type:
        return None
    if isinstance(analysis_type, list):
        candidates = [str(c).strip().lower() for c in analysis_type if str(c).strip()]
    else:
        candidates = [s.strip().lower() for s in str(analysis_type).split(",") if s.strip()]

    ac_keywords = {"ac_gain", "bandwidth", "unity_bandwidth", "phase_margin", "cmrr_tran", "cmrr"}
    tran_keywords = {"tran_gain", "thd_input_range", "cmrr_tran"}
    dc_keywords = {"output_swing", "offset", "icmr", "power"}

    for c in candidates:
        if c in ac_keywords:
            return "ac"
    for c in candidates:
        if c in tran_keywords:
            return "transient"
    for c in candidates:
        if c in dc_keywords:
            return "dc"

    # default fallback
    return None


def format_simulation_types(tool_data_list):
    """Format unique simulation types with robust fallbacks.

    If `simulation_type` is missing, try to infer it from `analysis_type`. If still missing, default to 'dc'.
    Returns a list of dicts like: {"name": "ac_simulation"}.
    """
    unique_sim_types = set()
    formatted_output = []
    for tool_data in tool_data_list:
        sim_type = tool_data.get("simulation_type")
        if not sim_type:
            sim_type = _infer_simulation_type_from_analysis(tool_data.get("analysis_type"))
        if not sim_type:
            sim_type = "dc"  # safe default

        sim_type = str(sim_type).lower()

        sim_name = f"{sim_type}_simulation"
        if sim_name not in unique_sim_types:
            unique_sim_types.add(sim_name)
            formatted_output.append({"name": sim_name})
    return formatted_output


def format_simulation_tools(tool_data_list):
    """Simulation tools are internal; we do not expose run_ngspice to the LLM."""

    return []


def format_analysis_types(tool_data_list):
    """Format all analysis types from all tool calls, with cmrr_tran and thd_input_range appended last.

    This function tolerates missing keys and accepts analysis_type as string or list.
    """
    formatted_output = []
    cmrr_thd_items = []

    for tool_data in tool_data_list:
        analysis_type = tool_data.get("analysis_type")
        if not analysis_type:
            # Nothing explicit — try to infer a reasonable analysis from simulation_type
            sim_type = tool_data.get("simulation_type") or _infer_simulation_type_from_analysis(tool_data.get("analysis_type"))
            if sim_type == "ac":
                # default ac analyses when nothing provided
                formatted_output.append({"name": "ac_gain"})
                continue
            elif sim_type == "transient":
                formatted_output.append({"name": "tran_gain"})
                continue
            else:
                # default dc analysis
                formatted_output.append({"name": "output_swing"})
                continue

        # If analysis_type exists, normalize it
        if isinstance(analysis_type, str):
            analyses = [a.strip().lower() for a in analysis_type.split(",") if a.strip()]
        elif isinstance(analysis_type, list):
            analyses = [str(a).strip().lower() for a in analysis_type if str(a).strip()]
        else:
            analyses = [str(analysis_type).strip().lower()]

        for analysis in analyses:
            if analysis in ["cmrr_tran", "thd_input_range"]:
                cmrr_thd_items.append({"name": analysis})
            else:
                formatted_output.append({"name": analysis})

    # Append cmrr and thd items at the end
    formatted_output.extend(cmrr_thd_items)
    return formatted_output


def combine_results(sim_types, sim_tools, analysis_types):
    """Combine simulation types, tools, and analysis types into one list."""
    return sim_types + sim_tools + analysis_types
