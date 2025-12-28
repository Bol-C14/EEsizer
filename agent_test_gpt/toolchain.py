"""Helpers for parsing and formatting tool calls."""

import json


def extract_tool_data(tool):
    """Extract tool call data from all tool calls in the tool object.

    Robust behavior and fallbacks:
    - Safely parse `function.arguments` whether it's JSON, a concatenated JSON objects string, or a dict.
    - If `simulation_tool` is missing, fallback to the tool-call function name or a sensible default ('run_ngspice').
    - Return a list of normalized dicts with keys: simulation_type, analysis_type, simulation_tool, raw_args.
    """
    message = tool.choices[0].message
    tool_data_list = []

    # Helper to normalize a single parsed dict and a tool_call object
    def _normalize(parsed_obj, tool_call_obj):
        if not isinstance(parsed_obj, dict):
            parsed_obj = {}
        sim_type = parsed_obj.get("simulation_type")
        analysis = parsed_obj.get("analysis_type")
        sim_tool = parsed_obj.get("simulation_tool")

        # Fallbacks for simulation_tool
        if not sim_tool:
            sim_tool = getattr(tool_call_obj.function, "name", None) \
                       or getattr(tool_call_obj, "name", None) \
                       or "run_ngspice"

        # Notebook schema uses a single wrapper tool name; map it to our local runner.
        if str(sim_tool).lower() == "universal_circuit_tool":
            sim_tool = "run_ngspice"

        return {
            "simulation_type": sim_type,
            "analysis_type": analysis,
            "simulation_tool": sim_tool,
            "raw_args": parsed_obj,
        }

    for tool_call in getattr(message, "tool_calls", []) or []:
        arguments = getattr(tool_call.function, "arguments", None)

        # If arguments is already a dict-like object
        if isinstance(arguments, dict):
            tool_data_list.append(_normalize(arguments, tool_call))
            continue

        # If arguments is a string, try several parse strategies
        parsed = None
        if isinstance(arguments, str):
            arg_str = arguments.strip()
            # Attempt JSON parse
            try:
                parsed = json.loads(arg_str)
            except Exception:
                # Try to coerce concatenated JSON objects into a list: '}{' -> '},{'
                try:
                    combined = "[" + arg_str.replace("}{", "},{") + "]"
                    arr = json.loads(combined)
                    if isinstance(arr, list):
                        # normalize each item
                        for item in arr:
                            tool_data_list.append(_normalize(item, tool_call))
                        continue
                except Exception:
                    parsed = None

                # As a last resort, try a safe eval (very limited)
                try:
                    parsed = eval(arg_str, {"__builtins__": {}}, {})
                except Exception:
                    parsed = None

        # If parsed successfully as a dict
        if isinstance(parsed, dict):
            tool_data_list.append(_normalize(parsed, tool_call))
        elif parsed is None:
            # No usable arguments -> still produce an entry with fallbacks
            tool_data_list.append(_normalize({}, tool_call))

    return tool_data_list


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
        candidates = analysis_type
    else:
        candidates = [s.strip() for s in str(analysis_type).split(",") if s.strip()]

    ac_keywords = {"ac_gain", "bandwidth", "unity_bandwidth", "phase_margin", "cmrr_tran", "cmrr"}
    tran_keywords = {"tran_gain", "thd_input_range", "cmrr_tran"}
    dc_keywords = {"output_swing", "offset", "ICMR", "power"}

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

        sim_name = f"{sim_type}_simulation"
        if sim_name not in unique_sim_types:
            unique_sim_types.add(sim_name)
            formatted_output.append({"name": sim_name})
    return formatted_output


def format_simulation_tools(tool_data_list):
    """Format unique simulation tools. Use fallbacks when fields are missing.

    Returns a list of dicts like: {"name": "run_ngspice"}.
    """
    unique_sim_tools = set()
    formatted_output = []
    for tool_data in tool_data_list:
        sim_tool = tool_data.get("simulation_tool") or tool_data.get("raw_args", {}).get("simulation_tool")
        if not sim_tool:
            # fallback to any sensible metadata contained in raw_args or default
            sim_tool = tool_data.get("raw_args", {}).get("tool") or tool_data.get("raw_args", {}).get("tool_name")
        if not sim_tool:
            sim_tool = "run_ngspice"

        if sim_tool not in unique_sim_tools:
            unique_sim_tools.add(sim_tool)
            formatted_output.append({"name": sim_tool})
    return formatted_output


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
            analyses = [a.strip() for a in analysis_type.split(",") if a.strip()]
        elif isinstance(analysis_type, list):
            analyses = analysis_type
        else:
            analyses = [str(analysis_type)]

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
