#%%
import json
import os
from agent_test_gpt import prompts
from agent_test_gpt import config
from agent_test_gpt.llm_client import (
    make_chat_completion_request,
    make_chat_completion_request_function,
)
from agent_test_gpt.toolchain import (
    extract_tool_data,
    format_simulation_types,
    format_simulation_tools,
    format_analysis_types,
    combine_results,
)
from agent_test_gpt.netlist_utils import (
    nodes_extract,
    extract_code,
    extract_number,
)
from agent_test_gpt.simulation_utils import (
    dc_simulation,
    ac_simulation,
    trans_simulation,
    out_swing,
    offset,
    ICMR,
    tran_gain,
    ac_gain,
    bandwidth,
    unity_bandwidth,
    phase_margin,
    stat_power,
    cmrr_tran,
    thd_input_range,
    filter_lines,
    convert_to_csv,
    format_csv_to_key_value,
    read_txt_as_string,
    run_ngspice,
)
from agent_test_gpt.optimization import (
    ToolCallingContext,
    ToolCallingDeps,
    ToolChainRunner,
    OptimizationContext,
    OptimizationDeps,
    OptimizationConfig,
    OptimizationRunner,
)
from agent_test_gpt.reporting import (
    run_multiple_optimizations,
    copy_netlist,
    plot_optimization_history,
)
#%%
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API URL and API key
#api_base_url = os.getenv('API_URL')
api_key = os.getenv('OPENAI_API_KEY')
#%% md
# ## User Input
#%%
"""
User input section, modify the netlist and tasks_generation_question as needed.
User must giving specific optimization targets in the tasks_generation_question and numbers.

TODO: so far also hardcoded, as user have to giving required numbers. It is ok to miss some requirements, but should not exceed the pre-defined specs that can be optimized.
"""


tasks_generation_question = " This is a circuit netlist, optimize this circuit with a output swing above 1.2V, input offset smaller than 0.001V, input common mode range bigger than 1.2, ac gain and transient gain above 65dB, unity bandwidth above 5000000Hz, phase margin bigger than 45 degree, power smaller than 0.05W, cmrr bigger than 100dB and thd small than -26dB"


"""
Chang customized defined netlist format.

Usually based on the sequence of nodes

M - mosfet, R - resistor, C - capacitor, L - inductor, V - voltage source, I - current source

must have .include for technology file
must have .op and .end at the end of netlist
"""

netlist = ''' 
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

'''
#%% md
# ## Task Decpmosition
#%%
## prompt for tasks generation based on user input

#task generation
tasks_generation_prompt = prompts.build_tasks_generation_prompt(tasks_generation_question, netlist)

tasks = make_chat_completion_request(tasks_generation_prompt) # json format plain text

#%%
type_question = ""

# set type_question, node_question, sim_question, sizing_question globally
def get_tasks(tasks):
    try:
        # Step 1: Remove triple backticks (if present)
        tasks = tasks.strip().strip('`')  # Remove leading/trailing backticks and whitespace
        tasks = tasks.replace("json", "").strip()
        # Step 2: Print the cleaned input for debugging
        print("Cleaned JSON string:")
        print(tasks)

        # Step 3: Parse the JSON string into a dictionary
        tasks_output = json.loads(tasks)

        # Step 4: Process the questions
        for question_dict in tasks_output["questions"]:
            for key, value in question_dict.items():
                #print(f"{key}: {value}")  # Print or process the key-value pairs
                # assign them to globals (not recommended unless necessary)
                globals()[key] = value


        ## key contains four stuff four different question, which naturally becomes four glpbal variables
        # type_question = "str"
        # node_question = "str"
        # sim_question = "str"
        # sizing_question = "str"
        # ----

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Invalid JSON string:", tasks)
    except KeyError as e:
        print(f"Missing key in JSON: {e}")

get_tasks(tasks)

#%%
target_value_prompt = prompts.build_target_value_prompt(tasks_generation_question)
target_values = make_chat_completion_request(target_value_prompt)

#%%
type_identify_prompt = prompts.build_type_identify_prompt(type_question, netlist)
print(type_identify_prompt)
#%%
type_identified = make_chat_completion_request(type_identify_prompt)
print(type_identified)

node_prompt = prompts.build_node_prompt(node_question, netlist)
#%%
nodes = make_chat_completion_request(node_prompt)
print(nodes)
#%%
input_nodes, output_nodes, source_names = nodes_extract(nodes)
print("----------------------node extract-----------------------------")
print(f"input_nodes:{input_nodes}")
print(f"output_nodes:{output_nodes}")
print(f"source_names:{source_names}")
#%%
Gain_init = None
Bw_init = None
UBw_init = None
Pm_init = None
CMRR_init = None
Power_init = None
Thd_init = None
OW_init = None
Offset_init = None
ICMR_init = None

"""
LLM can use this to decide which tool to call based on user input
here we have: dc_simulation, ac_simulation, transient_simulation, run_ngspice, ac_gain, tran_gain, output_swing, offset, icmr, bandwidth, unity_bandwidth, phase_margin, cmrr_tran, power, thd_input_range

"""
def tool_calling(tool_chain):
    global Gain_init, Bw_init, Pm_init, Dc_Gain_init, Tran_Gain_init, CMRR_init, Power_init, InputRange_Init, Thd_init, OW_init, Offset_init, UBw_init, ICMR_init

    context = ToolCallingContext(
        netlist=netlist,
        source_names=source_names,
        output_nodes=output_nodes,
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
    result = runner.run(tool_chain)

    metrics = result.metrics
    if metrics["gain"] is not None:
        Gain_init = metrics["gain"]
    if metrics["output_swing"] is not None:
        OW_init = metrics["output_swing"]
    if metrics["icmr"] is not None:
        ICMR_init = metrics["icmr"]
    if metrics["offset"] is not None:
        Offset_init = metrics["offset"]
    if metrics["tran_gain"] is not None:
        Tran_Gain_init = metrics["tran_gain"]
    if metrics["bandwidth"] is not None:
        Bw_init = metrics["bandwidth"]
    if metrics["unity_bandwidth"] is not None:
        UBw_init = metrics["unity_bandwidth"]
    if metrics["phase_margin"] is not None:
        Pm_init = metrics["phase_margin"]
    if metrics["cmrr"] is not None:
        CMRR_init = metrics["cmrr"]
    if metrics["power"] is not None:
        Power_init = metrics["power"]
    if metrics["thd"] is not None:
        Thd_init = metrics["thd"]
    if metrics["input_range"] is not None:
        InputRange_Init = metrics["input_range"]

    return result.sim_output, result.sim_netlist

print(sim_question)
#%%
print(sizing_question)
#%%
sim_prompts_gen = prompts.build_simulation_prompt(sizing_question)
print(sim_prompts_gen)
#sim_prompts = make_chat_completion_request(sim_prompts_gen)
#%%
tool = make_chat_completion_request_function(sizing_question)

# Example usage
# tool = ...  # Your tool object

# Step 1: Extract tool data from all tool calls
tool_data_list = extract_tool_data(tool)

# Step 2: Format simulation types, tools, and analysis types
formatted_sim_types = format_simulation_types(tool_data_list)
formatted_sim_tools = format_simulation_tools(tool_data_list)
formatted_analysis_types = format_analysis_types(tool_data_list)

# Step 3: Combine results
tool_chain_row = combine_results(formatted_sim_types, formatted_sim_tools, formatted_analysis_types)
tool_chain = {"tool_calls": tool_chain_row}
print("----------------------function used-----------------------------")
print(tool_chain)
sim_output, sim_netlist = tool_calling(tool_chain)
print("-------------------------result---------------------------------")
print(sim_output)
print("-------------------------netlist---------------------------------")
print(sim_netlist)
#%% md
# ## Optimization
#%%
sizing_Question = f"Currently, {sim_output}. " + sizing_question
print(sizing_Question)
#%%
def optimization(tools, target_values, sim_netlist, extracting_method):
    context = OptimizationContext(
        sim_output=sim_output,
        sizing_question=sizing_question,
        type_identified=type_identified,
    )

    def _missing_dc_gain(*_args, **_kwargs):
        raise NameError("name 'dc_gain' is not defined")

    deps = OptimizationDeps(
        make_chat_completion_request=make_chat_completion_request,
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
    config = OptimizationConfig()
    runner = OptimizationRunner(context, deps, config)
    initial_metrics = {
        "gain_output": Gain_init,
        "tr_gain_output": Tran_Gain_init,
        "output_swing_output": OW_init,
        "input_offset_output": Offset_init,
        "icmr_output": ICMR_init,
        "ubw_output": UBw_init,
        "pm_output": Pm_init,
        "pr_output": Power_init,
        "cmrr_output": CMRR_init,
        "thd_output": Thd_init,
    }
    return runner.run(tools, target_values, sim_netlist, extracting_method, initial_metrics)
#%%
results = run_multiple_optimizations(target_values, sim_netlist, tool_chain, extract_number, optimization)
#%%
source_file = 'output/netlist.cir'
copy_netlist(source_file, config.NETLIST_OUTPUT_PATH)
#%%
plot_optimization_history()
if __name__ == "__main__":
    # Quick self-test for the tool parsing + formatting fallbacks
    class FakeFunction:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, function):
            self.function = function

    class FakeMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, choices):
            self.choices = choices

    # Case A: full arguments
    f1 = FakeFunction(name="universal_circuit_tool", arguments='{"simulation_type": "ac", "analysis_type": "ac_gain", "simulation_tool": "run_ngspice"}')
    # Case B: missing simulation_tool
    f2 = FakeFunction(name="universal_circuit_tool", arguments='{"analysis_type": "tran_gain"}')
    # Case C: arguments is empty string
    f3 = FakeFunction(name="universal_circuit_tool", arguments='')
    # Case D: concatenated JSON objects (two tool calls encoded in one string)
    f4 = FakeFunction(name="universal_circuit_tool", arguments='{"analysis_type":"ac_gain"}{"analysis_type":"phase_margin"}')
    # Case E: arguments is None
    f5 = FakeFunction(name="universal_circuit_tool", arguments=None)

    tool_calls = [FakeToolCall(f) for f in [f1, f2, f3, f4, f5]]
    message = FakeMessage(tool_calls=tool_calls)
    response = FakeResponse(choices=[FakeChoice(message=message)])

    print("--- Running self-test for tool parsing ---")
    tlist = extract_tool_data(response)
    print("Extracted tool_data_list:")
    print(json.dumps(tlist, indent=2))

    sim_types = format_simulation_types(tlist)
    sim_tools = format_simulation_tools(tlist)
    analysis = format_analysis_types(tlist)

    print("\nFormatted simulation types:")
    print(sim_types)
    print("\nFormatted simulation tools:")
    print(sim_tools)
    print("\nFormatted analysis types:")
    print(analysis)

    # Build tool_chain and show
    tool_chain_row = combine_results(sim_types, sim_tools, analysis)
    tool_chain = {"tool_calls": tool_chain_row}
    print("\nCombined tool_chain:")
    print(json.dumps(tool_chain, indent=2))

    print("--- Self-test complete ---")
