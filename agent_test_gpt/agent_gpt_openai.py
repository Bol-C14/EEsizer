#%%
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from agent_test_gpt import config, prompts
from agent_test_gpt.llm_client import make_chat_completion_request, make_chat_completion_request_function
from agent_test_gpt.models import TaskQuestions
from agent_test_gpt.netlist_utils import extract_code, extract_number, nodes_extract, sanitize_netlist
from agent_test_gpt.optimization import (
    OptimizationConfig,
    OptimizationContext,
    OptimizationDeps,
    OptimizationRunner,
    ToolCallingContext,
    ToolCallingDeps,
    ToolChainRunner,
)
from agent_test_gpt.reporting import copy_netlist, plot_optimization_history, run_multiple_optimizations
from agent_test_gpt.simulation_utils import (
    ICMR,
    ac_gain,
    ac_simulation,
    bandwidth,
    cmrr_tran,
    convert_to_csv,
    dc_simulation,
    filter_lines,
    format_csv_to_key_value,
    out_swing,
    offset,
    phase_margin,
    read_txt_as_string,
    run_ngspice,
    stat_power,
    thd_input_range,
    tran_gain,
    trans_simulation,
    unity_bandwidth,
)
from agent_test_gpt.toolchain import (
    combine_results,
    extract_tool_data,
    format_analysis_types,
    format_simulation_tools,
    format_simulation_types,
    normalize_tool_chain,
    validate_tool_chain,
)
from agent_test_gpt.logging_utils import get_logger, setup_run_logging
from agent_test_gpt import llm_client

load_dotenv()
_logger = get_logger(__name__)

# Default sample inputs kept for backward compatibility; callers should pass their own.
DEFAULT_TASKS_GENERATION_QUESTION = (
    "This is a circuit netlist, optimize this circuit with a output swing above 1.2V, "
    "input offset smaller than 0.001V, input common mode range bigger than 1.2, "
    "ac gain and transient gain above 65dB, unity bandwidth above 5000000Hz, "
    "phase margin bigger than 45 degree, power smaller than 0.05W, "
    "cmrr bigger than 100dB and thd small than -26dB"
)

DEFAULT_NETLIST = """ 
.title Basic amp
.include 'ptm_90.txt'
.param Vcm = 0.6
M3 d3 in2 midp vdd pmos W=1u L=90n
M4 d4 in1 midp vdd pmos W=1u L=90n
M6 midp bias1 vdd vdd pmos W=1u L=90n

M1 d1 in1 midn 0 nmos W=1u L=90n
M2 d2 in2 midn 0 nmos W=1u L=90n
M5 midn bias2 0 0 nmos W=1u L=90n

M7 d1 g7 vdd vdd pmos W=1u L=90n
M8 d2 g7 vdd vdd pmos W=1u L=90n
M9 g7 bias3 d1 vdd pmos W=1u L=90n
M10 d10 bias3 d2 vdd pmos W=1u L=90n

M15 g7 bias5 s15 0 nmos W=1u L=90n
M16 s15 bias6 g7 vdd pmos W=1u L=90n
M17 d10 bias5 s17 0 nmos W=1u L=90n
M18 s17 bias6 d10 vdd pmos W=1u L=90n

M11 s15 bias4 d4 0 nmos W=1u L=90n
M12 s17 bias4 d3 0 nmos W=1u L=90n
M13 d4 s15 0 0 nmos W=1u L=90n
M14 d3 s15 0 0 nmos W=1u L=90n

M19 out d10 vdd vdd pmos W=1u L=90n
M20 out s17 0 0 nmos W=1u L=90n

Cc1 out d10 5p
Cc2 out s17 5p
Cl out 0 10p 
Rl out 0 1k

vbias1 bias1 0 DC 0.6
vbias2 bias2 0 DC 0.6
vbias3 bias3 0 DC 0.6
vbias4 bias4 0 DC 0.6
vbias5 bias5 0 DC 0.6
vbias6 bias6 0 DC 0.6

vdd vdd 0 1.2

Vcm cm 0 DC {Vcm}
Eidp cm in1 diffin 0 1
Eidn cm in2 diffin 0 -1
Vid diffin 0 AC 1 SIN (0 1u 10k 0 0)
.op

.end
"""


def _parse_json_payload(raw: str) -> dict:
    cleaned = raw.strip().strip("`").replace("json", "").strip()
    return json.loads(cleaned)


def parse_tasks_response(raw: str) -> TaskQuestions:
    payload = _parse_json_payload(raw)
    return TaskQuestions.from_json(payload)


def build_initial_metrics(metrics: dict) -> dict:
    """Map toolchain metrics into the optimization runner's expected initial fields."""
    return {
        "gain_output": metrics.get("gain"),
        "tr_gain_output": metrics.get("tran_gain"),
        "output_swing_output": metrics.get("output_swing"),
        "input_offset_output": metrics.get("offset"),
        "icmr_output": metrics.get("icmr"),
        "ubw_output": metrics.get("unity_bandwidth"),
        "pm_output": metrics.get("phase_margin"),
        "pr_output": metrics.get("power"),
        "cmrr_output": metrics.get("cmrr"),
        "thd_output": metrics.get("thd"),
    }


def tool_calling(tool_chain, netlist_text, source_names: List[str], output_nodes: List[str], output_dir: str):
    context = ToolCallingContext(
        netlist=netlist_text,
        source_names=source_names,
        output_nodes=output_nodes,
        output_dir=output_dir,
    )
    deps = ToolCallingDeps(
        dc_simulation=dc_simulation,
        ac_simulation=ac_simulation,
        trans_simulation=trans_simulation,
        run_ngspice=run_ngspice,
        filter_lines=filter_lines,
        convert_to_csv=convert_to_csv,
        format_csv_to_key_value=format_csv_to_key_value,
        read_txt_as_string=read_txt_as_string,
        ac_gain=ac_gain,
        out_swing=out_swing,
        ICMR=ICMR,
        offset=offset,
        tran_gain=tran_gain,
        bandwidth=bandwidth,
        unity_bandwidth=unity_bandwidth,
        phase_margin=phase_margin,
        cmrr_tran=cmrr_tran,
        stat_power=stat_power,
        thd_input_range=thd_input_range,
    )
    runner = ToolChainRunner(context, deps)
    return runner.run(tool_chain)


def run_agent(user_question: str, user_netlist: str, run_dir: str | None = None):
    """End-to-end agent run with explicit inputs instead of globals."""
    run_root = Path(run_dir or Path(config.RUN_OUTPUT_ROOT) / uuid.uuid4().hex)
    run_root.mkdir(parents=True, exist_ok=True)
    setup_run_logging(str(run_root), level=os.getenv("LOG_LEVEL"))
    manifest = {
        "user_question": user_question,
        "run_dir": str(run_root),
        "model": llm_client.DEFAULT_MODEL,
        "function_model": llm_client.DEFAULT_FUNCTION_MODEL,
        "temperature": llm_client.DEFAULT_TEMPERATURE,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cwd": os.getcwd(),
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    sanitized_netlist = sanitize_netlist(user_netlist)

    # Task decomposition
    tasks_generation_prompt = prompts.build_tasks_generation_prompt(user_question, sanitized_netlist)
    tasks_raw = make_chat_completion_request(tasks_generation_prompt)
    tasks_parsed = parse_tasks_response(tasks_raw)

    # Targets
    target_value_prompt = prompts.build_target_value_prompt(user_question)
    target_values = make_chat_completion_request(target_value_prompt)

    # Circuit type identification
    type_identify_prompt = prompts.build_type_identify_prompt(tasks_parsed.type_question, sanitized_netlist)
    _logger.debug("Type identify prompt: %s", type_identify_prompt)
    type_identified = make_chat_completion_request(type_identify_prompt)
    _logger.info("Type identified: %s", type_identified)

    # Node discovery
    node_prompt = prompts.build_node_prompt(tasks_parsed.node_question, sanitized_netlist)
    nodes_raw = make_chat_completion_request(node_prompt)
    _logger.info("Nodes response: %s", nodes_raw)
    input_nodes, output_nodes, source_names = nodes_extract(nodes_raw)
    if not output_nodes:
        raise ValueError("Node extraction returned no output_nodes; aborting run.")
    if not source_names:
        raise ValueError("Node extraction returned no source_names; aborting run.")
    _logger.info("input_nodes:%s output_nodes:%s source_names:%s", input_nodes, output_nodes, source_names)

    # Simulation tool chain from LLM
    sim_prompts_gen = prompts.build_simulation_prompt(tasks_parsed.sizing_question)
    _logger.info("Simulation prompt: %s", sim_prompts_gen)
    tool_response = make_chat_completion_request_function(tasks_parsed.sizing_question)
    tool_data_list = extract_tool_data(tool_response)
    formatted_sim_types = format_simulation_types(tool_data_list)
    formatted_sim_tools = format_simulation_tools(tool_data_list)
    formatted_analysis_types = format_analysis_types(tool_data_list)
    tool_chain_row = combine_results(formatted_sim_types, formatted_sim_tools, formatted_analysis_types)
    tool_chain = normalize_tool_chain({"tool_calls": tool_chain_row})
    validate_tool_chain(tool_chain)
    _logger.info("Tool chain: %s", tool_chain)

    tool_result = tool_calling(tool_chain, sanitized_netlist, source_names, output_nodes, output_dir=str(run_root))
    sim_output = tool_result.sim_output
    sim_netlist = tool_result.sim_netlist
    _logger.info("Initial simulation output: %s", sim_output)
    _logger.debug("Initial sim netlist: %s", sim_netlist)

    sizing_question_full = f"Currently, {sim_output}. " + tasks_parsed.sizing_question
    _logger.info("Sizing question: %s", sizing_question_full)
    opt_config = OptimizationConfig.from_output_dir(str(run_root))

    def optimization(tools, target_values_str: str, sim_netlist_text: str, extracting_method):
        context = OptimizationContext(
            sim_output=sim_output,
            sizing_question=tasks_parsed.sizing_question,
            type_identified=type_identified,
            source_names=source_names,
            output_nodes=output_nodes,
            output_dir=str(run_root),
        )

        def _missing_dc_gain(*_args, **_kwargs):
            _logger.warning("dc_gain called but not implemented; returning None.")
            return None

        deps = OptimizationDeps(
            make_chat_completion_request=make_chat_completion_request,
            sanitize_netlist=sanitize_netlist,
            dc_simulation=dc_simulation,
            ac_simulation=ac_simulation,
            trans_simulation=trans_simulation,
            run_ngspice=run_ngspice,
            filter_lines=filter_lines,
            convert_to_csv=convert_to_csv,
            format_csv_to_key_value=format_csv_to_key_value,
            read_txt_as_string=read_txt_as_string,
            ac_gain=ac_gain,
            dc_gain=_missing_dc_gain,
            out_swing=out_swing,
            offset=offset,
            ICMR=ICMR,
            tran_gain=tran_gain,
            bandwidth=bandwidth,
            unity_bandwidth=unity_bandwidth,
            phase_margin=phase_margin,
            stat_power=stat_power,
            thd_input_range=thd_input_range,
            cmrr_tran=cmrr_tran,
            extract_code=extract_code,
        )
        runner = OptimizationRunner(context, deps, opt_config)
        initial_metrics = build_initial_metrics(tool_result.metrics)
        return runner.run(tools, target_values_str, sim_netlist_text, extracting_method, initial_metrics)

    try:
        results = run_multiple_optimizations(target_values, sim_netlist, tool_chain, extract_number, optimization)
        source_file = Path(run_root) / "netlist.cir"
        final_copy = Path(run_root) / "netlist_final.cir"
        copy_netlist(str(source_file), str(final_copy))
        plot_optimization_history(csv_path=opt_config.csv_file, output_pdf=str(Path(run_root) / "optimization.pdf"))
        return results
    except Exception as exc:
        _logger.error("Run failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    run_agent(DEFAULT_TASKS_GENERATION_QUESTION, DEFAULT_NETLIST)
