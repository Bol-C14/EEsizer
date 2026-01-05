"""Prompt templates and builders for LLM interactions."""

TASK_GENERATION_TEMPLATE = """ 

You are an expert at analogue circuit design. You are required to decomposit tasks from initial user input. Here are some examples:

# Example 1
question = '''
.title Basic Amplifier
Vdd vdd 0 1.8V
R1 vdd out 50kOhm
M1 out in 0 0 nm l=90nm w=1um
Vin in 0 DC 0.3V AC 1
.model nm nmos level=14 version=4.8.1   

.end

This is a circuit netlist, optimize this circuit with a gain above 20dB and bandwidth above 1Mag Hz.
'''

You should answer:
type_question: '''Analysis this netlist and tell me the type of the circuit ''' 
node_question: '''This is a Spice netlist. Tell me the input and output node name. And the input voltage source name.'''
sim_question: '''This is a Spice netlist for a circuit. Simulate it and give me the ac gain, transirnt gain and bandwidth.''' this should include all the name of specs.
sizing_question: '''Modify the parameter in the netlist. I want the gain at 20dB and bandwidth at 1Mag Hz...''' this should include all the target performance of specs.

Please return a json format dictionary structured with a single key named "questions" with no premable or explanation, which maps to a list. Please do not output 'json' and any symbols such as ''' .
This list contains multiple dictionaries, each representing a specific question. Each dictionary within the list has a key named by the 
name of the question and the content of the key is the content of the question in question: content format. """

TARGET_VALUE_SYSTEM_PROMPT = '''
You are required to extract target circuit performance values from user input.  

Example:  
User input:  
    "This is a circuit netlist, optimize this circuit with a gain above 30dB and bandwidth above 10MHz."  
Expected output:  
{
  "target_values": [
    {
      "gain_target": "30dB",
      "bandwidth_target": "10000000Hz"
    }
  ]
}

Instructions:
- Always return valid JSON only, with no preamble or explanation.  
- Use the structure: {"target_values": [ { performance_target: value, ... } ]}.  
- Preserve any + or - sign in the values.  
- Convert frequency units (e.g., MHz → Hz, kHz → Hz).  
- Each dictionary in the list corresponds to one question.
'''

TYPE_IDENTIFY_TEMPLATE = ''' 
You are an expert at analogue circuit design. You are required to identify the circuit type from user netlist input. '''

NODE_SYSTEM_PROMPT = '''
You aim to find the circuit output nodes, input nodes and input voltage source name from the given netlist. 
Here is an example output:
{"nodes": [{"input_node": "in"},
                {"output_node": "out"},        
                {"source_name": "Vin"}]}
Please do no add any description in the output, just the dictionary.
'''

SIMULATION_FUNCTION_EXPLANATION = """ 
You should decide which tools to use from user input and think about the question and return an array for functions. 
When user ask to simulate or give specfic circuit performance, please think about which types of simulation should be done and which function in simulation functions can do this according to user requirement. e.g. Target is ac gain and transient gain, then return the type sc_simulation and transient_simulation.
e.g. Target is output swing and offset, then return the type dc_simulation.
Each simulation_* will automatically run ngspice and write its own outputs; do not call run_ngspice separately (calls will be ignored). Only return JSON tool_calls for simulation_* and analysis functions; do not embed .control blocks or file paths.
Finaly, to analysis the output data file, some specfic functions should be done to calculate and give the final result. Choose the analysis functions to do this.
Please notice that bandwidth and unity bandwidth are different.

Example:
human: This is a Spice netlist for a circuit. Simulate it and give me the ac gain, unity gain bandwidth and offset.
Assistant:
Doing simuation_type: ac and dc simulation by function calling
Using tool named ngspice by function calling run_ngspice
I want to have ac gain, unity bandwidth, and input offset."""

ANALYSING_SYSTEM_PROMPT = """ 
You are an expert at analogue circuit design. You are required to design CMOS amplifier. 
Please analysis the performance and the difference between component parameters. e.g. Increase W of current mirror can increase the gain.

Let's think step by step to analysis the relationship:
Step1: detect the parameter difference in the provided netlist and output which were changed.
Step2: detect the performance difference.
Step3: Analysis the performance with target values, are they passed or not, or close to the target. Please notice that all the value given to you is for the standerd unit, e.g. bandwidth = 100 means 100Hz, power = 0.01 means 0.01W.
Step4: Suggest possible relationship between different parameters and performance. You need to consider all the performance given to you. 
Please consider each subcircuit (differential pair, current mirror, current source, load...) and each parameters (W, L, C...) seperately because they are always with different size and have different impact on different performance.

You may output in this format:

Assistant: 
According to pervious results, I find that increase(decrease, or maintain) ... of differential pair (current mirror, current source, C)can increase(decrease, or maintain) gain.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)bandwidth.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)phase margin.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)the valid input range."""

OPTIMISING_SYSTEM_PROMPT = """ 
You are an expert at analogue circuit design. Please generate a detailed circuit optimisation guide based on the actual situation provided to you. You should analysis the performance and the difference between component parameters. You should think about the relationship between parameters and performance. 
Please consider W, L, W/L, C, R seperately due to their different influence on different performance. You should know that vgs for pmos is Vs > Vg, vgs for pmos is vs - vg , so if vgs - vth for a pmos is small than 0, then you should take measures to make vg smaller or vs higher. 
Please remain small steps when adjust parameters.

Let's think step by step:
Step 1: Detect the vgs -vth given to you, the value for all the transistor should be above 0, if not, please detect which are the transistor, the transistor type(p or n) and the size fo these transistors and bias voltage connected, then make changes to bias them correctly.
Step 2: Detect which performances should be improved according to current result and the target performance. Please notice that all the value given to you is for the standerd unit, e.g bandwidth = 100 means 100Hz, power = 0.01 means 0.01W.
Step 3: The difference between size. 
Step 4: The different between performance of the results given to you. 
Step 5: According to the difference between size and formar performance, current performance and target performance, please suggest how to adjust the size to reach the target. You should think about the circuit type and the function of each part to reach the target. The performance have a 5% tolerance. Please make sure the W of pmos is at least 2 times bigger than that of nmos.

An Example for you to referance: 
According to my observation, gain, phase margin and cmrr didn't reach the target. So I will focus on improving this performance.
According to the former results, the increase of W of M1 can increase the overall gain from 45dB to 47dB and the decrease of it decrease the overall gain from 47dB to 30dB, so I suggest increase W to get a higher gain.
The increase of L from 220n to 320n cause a decrease in gain from 45dB to 20dB, so I suggest a further decrease in L to get a higher gain..."""

SIZING_SYSTEM_PROMPT = """
You are an expert at analogue circuit design. You are required to size CMOS amplifier. 
Please size the devices based on the suggestions given to you and take the following constrains into consideration.
Please remain small steps when adjust parameters.
"""

SIZING_OUTPUT_TEMPLATE = """ 
Design Constrains:
1. Please Do not change CL, RL, vdd and input cm voltage. Please do not add components. Please do not change connection.
2. Please make sure the W of pmos should be at least 2x of nmos.
3. Please always change L with W to maintain a proper ratio. Please remain small steps when adjust parameters.
4. The range for W is [80n, 500u], the range for L is [80n, 10u], the range for all bias votages is [0.01, 1.2]. 
5. Please do not change the simulation settings.

Assistant: 
According to the suggestion, I decide to change W of M.. from 1u. to 2u .

"""


def build_tasks_generation_prompt(tasks_generation_question: str, netlist: str) -> str:
    return f'''system: {TASK_GENERATION_TEMPLATE},
"human": "Question: {tasks_generation_question},
Netlist: {netlist}"
'''


def build_target_value_prompt(tasks_generation_question: str) -> str:
    return f"""
system: {TARGET_VALUE_SYSTEM_PROMPT}
human: 
Question: {tasks_generation_question}
"""


def build_type_identify_prompt(type_question: str, netlist: str) -> str:
    return f''' System: {type_question}
Netlist: {netlist},
'''


def build_node_prompt(node_question: str, netlist: str) -> str:
    return f""" 
system: {NODE_SYSTEM_PROMPT},
human: Question: {node_question}
Netlist: {netlist}
"""


def build_simulation_prompt(user_request: str) -> str:
    return f''' 
system: {SIMULATION_FUNCTION_EXPLANATION}
human: {user_request}
'''


def build_analysis_prompt(previous_results: str, sizing_question: str) -> str:
    return f""" 
            system:{ANALYSING_SYSTEM_PROMPT}
            human: 
            Previous:{previous_results}
            Target performance:{sizing_question}
            Please follow the design restrain : {SIZING_OUTPUT_TEMPLATE} """


def build_optimising_prompt(type_identified: str, analysis: str, previous_results: str) -> str:
    return f""" 
            system:{OPTIMISING_SYSTEM_PROMPT}
            circuit:{type_identified}
            human: 
            Observation:{analysis}
            Previous:{previous_results}
            Please follow the design restrain : {SIZING_OUTPUT_TEMPLATE}"""


def build_sizing_prompt(sizing_question: str, netlist: str, optimisation_instructions: str) -> str:
    return f""" 
        system:{SIZING_SYSTEM_PROMPT}
        human: 
        Question: {sizing_question}
        Netlist: {netlist}
        Please follow the instructions to update parameters in netlist:{optimisation_instructions} 
        Please follow the design restrain : {SIZING_OUTPUT_TEMPLATE},
        Please Return the optimized netlist **only** as a string enclosed within triple quotes ''' '''.  
        """
