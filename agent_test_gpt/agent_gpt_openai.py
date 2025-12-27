#%%
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import boto3
import json
import os
import csv
import time
import shutil
import pandas as pd
from pathlib import Path
from scipy.fft import fft, fftfreq
from agent_test_gpt import prompts
from agent_test_gpt.llm_client import (
    make_chat_completion_request,
    make_chat_completion_request_function,
)
#%%
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API URL and API key
#api_base_url = os.getenv('API_URL')
api_key = os.getenv('OPENAI_API_KEY')


_INCLUDE_PATTERN = re.compile(r"^\s*\.include\s+['\"]?(?P<path>[^'\"\s]+)['\"]?", re.IGNORECASE | re.MULTILINE)


def _resolve_spice_include_path(raw_path: str) -> str | None:
    """Resolve a relative .include path against known project folders.

    This keeps notebook-derived netlists (e.g. `.include 'ptm_90.txt'`) runnable
    when executed from the repository root.
    """

    include_path = Path(raw_path)
    if include_path.is_absolute():
        return raw_path if include_path.exists() else None

    cwd = Path.cwd()
    if (cwd / include_path).exists():
        return raw_path

    search_roots = (
        cwd / "agent_test_gpt",
        cwd / "agent_test_gemini",
        cwd / "agent_test_claude",
        cwd / "legacy_notebook" / "agent_test_gpt",
        cwd / "legacy_notebook" / "agent_test_gemini",
        cwd / "legacy_notebook" / "agent_test_claude",
        cwd / "variation",
    )
    for root in search_roots:
        candidate = root / include_path
        if candidate.exists():
            try:
                return str(candidate.relative_to(cwd))
            except ValueError:
                return str(candidate)
    return None


def normalize_spice_includes(netlist_text: str) -> str:
    """Rewrite `.include` lines to point at resolvable files when possible."""

    def _repl(match: re.Match[str]) -> str:
        raw_path = match.group("path")
        resolved = _resolve_spice_include_path(raw_path)
        if not resolved or resolved == raw_path:
            return match.group(0)
        return match.group(0).replace(raw_path, resolved, 1)

    return _INCLUDE_PATTERN.sub(_repl, netlist_text)

#print(f"API URL: {api_base_url}")
#print(f"API Key: {api_key}")
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
#%%
"""
extracting node names from LLM output
asked the LLM to
"""

def nodes_extract(node):
    try:
        # Step 1: Clean the input string (remove backticks, "json", and extra whitespace)
        nodes = node.strip().strip('`')  # Remove leading/trailing backticks and whitespace
        nodes = nodes.replace("json", "").strip()  # Remove the word "json" and any extra whitespace

        # Step 2: Parse the JSON string into a dictionary
        node_name = json.loads(nodes)  # Use json.loads() for parsing strings

        # Step 3: Initialize lists to store input nodes, output nodes, and source names
        input_nodes = []
        output_nodes = []
        source_names = []

        # Step 4: Iterate through the "nodes" list (support multiple schema variants)
        for node_item in node_name.get("nodes", []):
            if not isinstance(node_item, dict):
                continue

            # Legacy format: {"input_node": "in1"}, {"source_name": "Vid"}
            if "input_node" in node_item:
                input_nodes.append(node_item["input_node"])
                continue
            if "output_node" in node_item:
                output_nodes.append(node_item["output_node"])
                continue
            if "source_name" in node_item:
                source_names.append(node_item["source_name"])
                continue

            # Newer format: {"input_nodes": [..]}, {"source_names": [..]}
            if "input_nodes" in node_item and isinstance(node_item["input_nodes"], list):
                input_nodes.extend([str(x) for x in node_item["input_nodes"]])
                continue
            if "source_names" in node_item and isinstance(node_item["source_names"], list):
                source_names.extend([str(x) for x in node_item["source_names"]])
                continue

        # Step 5: Return the extracted lists
        return input_nodes, output_nodes, source_names

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return [], [], []
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
        return [], [], []

def extract_code(text):
    regex1 = r"'''(.+?)'''" 
    regex2 = r"```(.+?)```"

    matches1 = re.findall(regex1, text, re.DOTALL)
    matches2 = re.findall(regex2, text, re.DOTALL)

    extracted_code = "\n".join(matches1 + matches2)
    lines = extracted_code.split('\n')
    cleaned_lines = []

    for line in lines:
        if '*' in line:
            line = line.split('*')[0].strip()
        elif '#' in line:
            line = line.split('#')[0].strip()
        elif ';' in line:
            line = line.split(';')[0].strip()
        elif line.startswith('verilog'):
            line = '\n'
        if line:  
            cleaned_lines.append(line)
    
    cleaned_code = "\n".join(cleaned_lines)
    return cleaned_code

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
#fuctions for different simulation type
def dc_simulation(netlist, input_name, output_node):
     end_index = netlist.index('.end\n')

     #input_nodes_str = ' '.join(input_name)
     output_nodes_str = ' '.join(output_node)

     simulation_commands = f'''
    
    .control
      dc Vcm 0 1.2 0.001        
      wrdata output/output_dc.dat {output_nodes_str}  
    .endc
     '''
     new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
     print(f"dc netlist:{new_netlist}")
     return new_netlist

def ac_simulation(netlist, input_name, output_node):
     end_index = netlist.index('.end\n')

     output_nodes_str = ' '.join(output_node)
     simulation_commands = f'''
      .control
        ac dec 10 1 10G        
        wrdata output/output_ac.dat {output_nodes_str} 
      .endc
     '''
     new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
     return new_netlist

def trans_simulation(netlist, input_name, output_node):
    end_index = netlist.index('.end\n')
    output_nodes_str = ' '.join(output_node)
    simulation_commands = f'''
      .control
        tran 50n 500u
        wrdata output/output_tran.dat {output_nodes_str} I(vdd) in1
      .endc
     '''
    new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
    return new_netlist

def tran_inrange(netlist):
    import re
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist, flags=re.DOTALL)
    netlist_set = ""
    for line in modified_netlist.splitlines():
        if line.startswith("Vid"):
            # Append AC 1 to the Vcm line
            netlist_set += "Vid diffin 0 AC 1 SIN (0 10u 10k 0 0)\n"
        else:
            # Keep other lines unchanged
            netlist_set += line + "\n"

    end_index = netlist_set.index('.end\n')
    simulation_commands = f'''
    .control
      set_numthread = 8
      let start_vcm = 0
      let stop_vcm = 1.25
      let delta_vcm = 0.05
      let vcm_act = start_vcm
      rm output/output_tran_inrange.dat

      while vcm_act <= stop_vcm
        alter Vcm vcm_act
        tran 50n 500u  
        wrdata output/output_tran_inrange.dat out in1 in2 cm 
        set appendwrite
        let vcm_act = vcm_act + delta_vcm 
      end
    .endc
     '''
    new_netlist = netlist_set[:end_index] + simulation_commands + netlist_set[end_index:]
    return new_netlist
#%%
from scipy.signal import find_peaks
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

def out_swing(filename):
    #vdd=1.2
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("Vcm"):
            line = 'Vin1 in1 0 DC {Vcm}'
        elif line.startswith("Eidn"):
            line = """Vin2 in2 0 DC 0.6\nR1 in in1 100k\nR2 in1 out 1000k"""
        elif line.startswith("Eidp"):
            line = '\n'
        elif line.startswith("Vid"):
            line = '\n'
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)
    simulation_commands = f'''
    .control
      dc Vin1 0 1.2 0.00005
      wrdata output/output_dc_ow.dat out in1
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_ow = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]
    print(netlist_ow)
    with open('output/netlist_ow.cir', 'w') as f:
        f.write(netlist_ow)
    run_ngspice(netlist_ow,'netlist_ow' )
    data_dc = np.genfromtxt(f'output/{filename}_ow.dat', skip_header=1)
    output = data_dc[0:,1]
    in1 = data_dc[0:,3]
    d_output_d_in1 = np.gradient(output, in1)

    # Replace zero or near-zero values with a small epsilon to avoid log10(0) error
    epsilon = 1e-10
    d_output_d_in1 = np.where(np.abs(d_output_d_in1) < epsilon, epsilon, np.abs(d_output_d_in1))
    print(d_output_d_in1)

    # Compute gain safely
    #gain = 20 * np.log10(np.abs(d_output_d_in1))
    grad = 10 
    indices = np.where(d_output_d_in1 >= 0.95*grad)
    output_above_threshold = output[indices]
    #print(output_above_threshold)
    if output_above_threshold.size > 0:
        vout_first = output_above_threshold[0]  # First value
        vout_last = output_above_threshold[-1]  # Last value
        ow = vout_first - vout_last # Difference

        print(f"First output: {vout_first}")
        print(f"Last output: {vout_last}")
        #print(f"Difference: {vout_diff}")
    else:
        ow = 0
        print("No values found where gain >= 0.9*grad")
    print(f'output swing: {ow}')
    return ow

def offset(filename):
    vdd=1.2
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("M4") or line.startswith("M1"):
            line = re.sub(r'\bin1\b', 'out', line)  # Replace 'in1' with 'out'
        elif line.startswith("Vcm"):
            line = 'Vin2 in2 0 DC 0.6'
        elif line.startswith("Eidn"):
            line = '\n'
        elif line.startswith("Eidp"):
            line = '\n'
        elif line.startswith("Vid"):
            line = '\n'
        #if not (line.startswith("Rl") or line.startswith("Cl")):  # Skip lines starting with "Rl" or "Cl"
        updated_lines.append(line)

        #updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)
    simulation_commands = f'''
    .control
      dc Vin2 0 1.2 0.0001
      wrdata output/output_dc_offset.dat out
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_offset = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]
    #print(netlist_offset)
    with open('output/netlist_offset.cir', 'w') as f:
        f.write(netlist_offset)
    run_ngspice(netlist_offset,'netlist_offset' )
    data_dc = np.genfromtxt(f'output/{filename}_offset.dat', skip_header=1)
    
    # Extract input and output values from the data
    input = data_dc[19:-19, 0]   # Skip first and last points
    output = data_dc[19:-19, 1]
    #print(input)
    input_index = np.where(input==0.6)
    output_offset = output[input_index]

    # Calculate the maximum voltage offset (difference between output and input)
    #voff= np.max(np.abs(output - input))
    if isinstance(output_offset, np.ndarray) and output_offset.size > 0:
        voff = np.abs(output_offset[0] -0.6)  # Take the first element if it's an array
    else:
        voff = float(voff)  # Convert to float if it's already a single value

    print(voff)
    
    return voff

def ICMR(filename):
    vdd=1.2
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()

    # Remove control block
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)

    # Update transistor connections
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("M4") or line.startswith("M1"):
            line = re.sub(r'\bin1\b', 'out', line)  # Replace 'in1' with 'out'
        #if not (line.startswith("Rl") or line.startswith("Cl")):  # Skip lines starting with "Rl" or "Cl"
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)

    # Append simulation commands
    simulation_commands = '''
    .control
      dc Vcm 0 1.2 0.001
      wrdata output/output_dc_icmr.dat out I(vdd)
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_icmr = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]

    # Write modified netlist to a new file and run simulation
    with open('output/netlist_icmr.cir', 'w') as f:
        f.write(netlist_icmr)

    run_ngspice(netlist_icmr, 'netlist_icmr')

    # Read simulation data
    data_dc = np.genfromtxt(f'output/{filename}_icmr.dat', skip_header=1)

    # Extract relevant data
    input_vals = data_dc[:, 0]   # Skip first and last points
    output_vals = data_dc[:, 1]
    
    gradient = np.gradient(output_vals, input_vals)
    vos = np.abs(output_vals - input_vals)

    # Find the index of gradient where equals 1.

    unit_gain_indices = np.where(gradient>=0.95)[0]
    vos_indices = np.where(vos <= 0.02)[0]
    #print(unit_gain_indices)  # Using a small tolerance
    if len(vos_indices) > 1:
        unit_gain_index1 = vos_indices[0]  # Take the first occurrence
        unit_gain_index2 = vos_indices[-1]  # # Last crossing point
        
        ic_min_grad = input_vals[unit_gain_index1]
        ic_min_voff = input_vals[vos_indices[0]]
        ic_max_voff = input_vals[vos_indices[-1]]

        ic_max_grad = input_vals[unit_gain_index2]
        ic_min = max(ic_min_grad, ic_min_voff)
        ic_max = min(ic_max_grad, ic_max_voff)
        icmr_out = ic_max - ic_min
    # Verify we have a proper range
    elif len(unit_gain_indices) == 1:
        icmr_out = 0
        print("Warning: Only one unit gain point found")

    else:
        print("Warning:no unit gain point found")
        icmr_out = 0  # No valid range found

    print(icmr_out)
    #print(f'ic_min = {ic_min}')
    #print(f'ic_max = {ic_max}')

    return icmr_out

def tran_gain(file_name):
    data_tran = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_tran.shape[1]

    # for one output node
    if num_columns == 6:
        time = data_tran[:, 0]
        out = data_tran[:, 1]

    # Find the peaks (local maxima)
        peak= np.max(out)

    # Find the troughs (local minima) by inverting the signal
        trough = np.min(out)

    # Compute the gain using the difference between average peak and average trough
        tran_gain = 20 * np.log10(np.abs(peak - trough)/0.000002)
    else:
        raise ValueError("The input file must have 2 columns.")
    
    print(f"tran gain = {tran_gain}")
    
    return tran_gain

def ac_gain(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]

    # for one output node
    if num_columns == 3:
        frequency = data_ac[:, 0]
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:    
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        gain = 20 * np.log10(np.abs(v_d[0]))
    else:
        raise ValueError("The input file must have either 3 or 6 columns.")
    
    print(f"gain = {gain}")
    
    return gain

def bandwidth(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:, 0]
    
    # for one output node
    if num_columns == 3:    
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))

    half_power_point = gain - 3
    
    indices = np.where(output >= half_power_point)[0]
    
    if len(indices) > 0:
        f_l = frequency[indices[0]]
        f_h = frequency[indices[-1]]
        bandwidth = f_h - f_l
    else:
        f_l = f_h = bandwidth = 0

    print(f"bandwidth = {bandwidth}")
    
    return bandwidth

def unity_bandwidth(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:, 0]
    
    # for one output node
    if num_columns == 3:    
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))

    half_power_point = 0
    
    indices = np.where(output >= half_power_point)[0]
    
    if len(indices) > 0:
        f_l = frequency[indices[0]]
        f_h = frequency[indices[-1]]
        bandwidth = f_h - f_l
    else:
        f_l = f_h = bandwidth = 0

    print(f"unity bandwidth = {bandwidth}")
    
    return bandwidth

def phase_margin(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:,0]
    # for one output node
    if num_columns == 3:
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]   
    # for 2 output nodes
    elif num_columns == 6:    
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
    #gain
    gain_db = 20 * np.log10(np.abs(v_d))
    #phase
    phase = np.degrees(np.angle(v_d))

    #find the frequency where gain = 0dB
    gain_db_at_0_dB = np.abs(gain_db - 0)
    index_at_0_dB = np.argmin(gain_db_at_0_dB)
    frequency_at_0_dB = frequency[index_at_0_dB]
    phase_at_0_dB = phase[index_at_0_dB]

    initial_phase = phase[0]
    tolerance = 15
    if np.isclose(initial_phase, 180, atol=tolerance):
        return phase_at_0_dB
    elif np.isclose(initial_phase, 0, atol=tolerance):
        return 180 - np.abs(phase_at_0_dB)
    else:
        return 0

def calculate_static_current(simulation_data):
    static_currents = []
    threshold=5e-7
    # calculate the difference of two time points
    for i in range(len(simulation_data)):
        current_diff = np.abs(simulation_data[i] - simulation_data[i-1])        
        if current_diff <= threshold:
            static_currents.append(simulation_data[i])

    if static_currents:
        return np.mean(static_currents)
    else:
        return None

def stat_power(filename, vdd=1.8):
    
    data_trans = np.genfromtxt(f'output/{filename}.dat')
    num_columns = data_trans.shape[1]
    if num_columns == 3:
        iout = data_trans[:, 3]  
        Ileak = calculate_static_current(iout)
        static_power = Ileak * vdd
    
    if num_columns == 6:
        iout = data_trans[:, 3]  
        Ileak = calculate_static_current(iout)
        static_power = np.abs(Ileak * vdd)

    print(f"power = {static_power}")

    return static_power
        
def cmrr_tran(netlist):
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()

    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)
    updated_lines = []

    for line in modified_netlist.splitlines():
        if line.startswith("Vcm"):
            line = 'Vin1 in1 out DC {Vcm} AC 1'
        elif line.startswith("Eidn"):
            line = """Vin2 in2 0 DC {Vcm} AC 1 """
        elif line.startswith("Eidp"):
            line = '\n'
        elif line.startswith("Vid"):
            line = '\n'
        #if not (line.startswith("Rl") or line.startswith("Cl")):  # Skip lines starting with "Rl" or "Cl"
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)

    simulation_commands = f'''
    .control
      set_numthread = 8
      let start_vcm = 0
      let stop_vcm = 1.25
      let delta_vcm = 0.05
      let vcm_act = start_vcm

      while vcm_act <= stop_vcm
        alter Vin1 vcm_act
        alter Vin2 vcm_act
        ac dec 10 1 10G
        wrdata output/output_inrange_cmrr.dat out
        set appendwrite
        let vcm_act = vcm_act + delta_vcm 
      end
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_cmrr = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]
    print(netlist_cmrr)
    with open('output/netlist_cmrr.cir', 'w') as f:
        f.write(netlist_cmrr)
    run_ngspice(netlist_cmrr,'netlist_cmrr' )
    data_ac = np.genfromtxt('output/output_inrange_cmrr.dat')
    freq = data_ac[:, 0]
    output = data_ac[:, 1] + 1j * data_ac[:, 2]
    # Find indices where freq = 10 GHz (end of a block)
    #block_ends = np.where(freq == 10e9)[0]
    #print(block_ends)

    idx_1000 = np.where(freq == 10000)[0]
    #print(len(idx_1000))

    vcm_values = np.arange(0, 1.2 + 0.05, 0.05)  # include the stop value
    #print(vcm_values)
    #print(len(vcm_values))

    out_1000 = output[idx_1000]
    cmrr_val = 20*np.log(np.abs(1 / out_1000))
    cmrr_ac = np.min(cmrr_val)
    cmrr_ac_max = np.max(cmrr_val)

    return cmrr_ac,cmrr_ac_max

def thd_input_range(filename):
    thd_values = []
    valid_inputs = []
    threshold_thd = -24.7
    #read origin netlist
    with open('output/netlist.cir', 'r') as file:
      netlist0 = file.read()
    #replace the simulation setting 
    netlist_inrange = tran_inrange(netlist0)
    #print(netlist_inrange)
    run_ngspice(netlist_inrange, 'netlist_inrange')

    #data preperation
    data_tran = np.genfromtxt(f'output/{filename}_inrange.dat')
    time = data_tran[:,0]
    other_data = data_tran[:, 1:]  # Extract other columns
    iteration_indices = np.where(time == 0)[0]
    batch_numbers = np.zeros_like(time, dtype=int)
    # Assign batch numbers based on iteration resets
    for i, idx in enumerate(iteration_indices):
      batch_numbers[idx:] = i
    # Create a DataFrame with batch information
    columns = ['time', 'batch'] + [f'col_{i}' for i in range(1, other_data.shape[1] + 1)]
    data_with_batches = np.column_stack((time, batch_numbers, other_data))
    df = pd.DataFrame(data_with_batches, columns=columns)

    #fft
    #plt.figure(figsize=(12, 8))
    for batch, group in df.groupby('batch'):
      time = group['time'].reset_index(drop=True).to_numpy()  
      #print(time)
      output = group['col_1'].reset_index(drop=True).to_numpy()    
      output_nodc = output - np.mean(output)
      #print(output)
      # Calculate the sampling frequency (Fs)
      time_intervals = 5e-8
      fs = 1 / time_intervals  # Sampling frequency in Hz
    
      N = len(output)  # Length of the signal
      fft_values = fft(output_nodc)
      fft_magnitude = np.abs(fft_values[:N//2])  # Take magnitude of FFT values (only positive frequencies)
      fft_freqs = fftfreq(N, d=1/fs)[:N//2]  # Corresponding frequency values (only positive frequencies)

      # Identify the fundamental frequency (largest peak)
      fundamental_idx = np.argmax(fft_magnitude)
      fundamental_freq = fft_freqs[fundamental_idx]
      fundamental_amplitude = fft_magnitude[fundamental_idx]

      # Calculate harmonic amplitudes (sum magnitudes of multiples of the fundamental frequency)
      harmonics_amplitude = 0
      for i in range(2, N // fundamental_idx):  # Start from second harmonic
          idx = i * fundamental_idx
          if idx < len(fft_magnitude):  # Ensure the index is within bounds
              harmonics_amplitude = harmonics_amplitude + fft_magnitude[idx] ** 2

      harmonics_rms = np.sqrt(harmonics_amplitude)

      # Calculate Total Harmonic Distortion (THD)
      if fundamental_amplitude == 0:
          fundamental_amplitude = 1e-6
          
      thd_db = 20 * np.log10(harmonics_rms / fundamental_amplitude)

      thd_values.append(thd_db)
    
      if thd_db < threshold_thd:
            valid_inputs.append(np.max(group['col_7']))

    thd = np.max(thd_values)
    print(thd)
    print(valid_inputs)

    if not valid_inputs:  # Check if valid_inputs is empty
        input_ranges = [(0, 0)]  # Return default range if no valid inputs
    
    else:
        input_ranges = []  # List to store the ranges
        start = valid_inputs[0]  # Start of the current range

        for i in range(1, len(valid_inputs)):
            if valid_inputs[i] - valid_inputs[i - 1] > 0.11:
            # If the difference exceeds the threshold, close the current range
                input_ranges.append((start, valid_inputs[i - 1]))
                start = valid_inputs[i]  # Start a new range

        # Add the last range
        input_ranges.append((start, valid_inputs[-1]))

    print(input_ranges)
    
    return thd, input_ranges

def is_range_covered(outer_range, sub_ranges):
    """
    Check if an outer range is fully covered by a list of subranges.

    Args:
        outer_range (tuple): The outer range as (start, end).
        sub_ranges (list of tuples): A list of subranges as (start, end).

    Returns:
        bool: True if the outer range is fully covered by the subranges, False otherwise.
    """
    # Sort subranges by their start values
    sub_ranges = sorted(sub_ranges)
    
    # Check if the outer range is fully covered
    current_position = outer_range[0]  # Start from the beginning of the outer range
    
    for sub_range in sub_ranges:
        # If there's a gap between current position and the start of the subrange, it's not covered
        if sub_range[0] > current_position:
            return False
        
        # Extend the current covered position if the subrange extends it
        if sub_range[1] > current_position:
            current_position = sub_range[1]
        
        # If the current position exceeds the end of the outer range, it's fully covered
        if current_position >= outer_range[1]:
            return True
    
    # If we exit the loop and haven't reached the end of the outer range, it's not covered
    return False

def filter_lines(input_file, output_file):
    
    filtered_lines = []
    with open(input_file, "r") as infile:
        for line in infile:
            stripped_line = line.lstrip()  # Remove leading whitespace
            words = line.split()
            if (stripped_line.startswith("device") and len(words) > 1 and words[1].startswith('m')) or stripped_line.startswith("vth ") or stripped_line.startswith("vgs "):
                filtered_lines.append(line.strip())  # Store non-empty lines without leading/trailing spaces
    with open(output_file, "w") as outfile:
        outfile.write("\n".join(filtered_lines))  # Write all lines at once   

def convert_to_csv(input_txt, output_csv):

    headers = []
    vth_rows = []
    vgs_rows = []

    with open(input_txt, "r") as infile:
        for line in infile:
            stripped_line = line.strip()

            # If the line starts with 'device', extract the device names (excluding the word 'device')
            if stripped_line.startswith("device"):
                devices = stripped_line.split()[1:]  # Skip the word 'device'
                headers.append(devices)
        # If the line starts with 'gm', extract the gm values
            elif stripped_line.startswith("vth"):
                vth_values = stripped_line.split()[1:]  # Skip the word 'gm'
                if vth_values:  # Ensure there are gm values to add
                    vth_rows.append(vth_values)

            elif stripped_line.startswith("vgs"):
                vgs_values = stripped_line.split()[1:]  # Skip the word 'gm'
                if vgs_values:  # Ensure there are gm values to add
                    vgs_rows.append(vgs_values)
            #rows = [item for sublist in rows for item in sublist]
        #print(headers)
        vth_rows = [float(item) for sublist in vth_rows for item in sublist]
        vgs_rows = [float(item) for sublist in vgs_rows for item in sublist]
        headers = [str(item) for sublist in headers for item in sublist]

    num_columns = len(headers)
    if num_columns <= 0:
        # Nothing to write; ngspice may have failed or the expected OP dump format isn't present.
        with open(output_csv, "w", newline="") as outfile:
            csv.writer(outfile).writerow([])
        return False

    vth_rows_2d = [vth_rows[i:i + num_columns] for i in range(0, len(vth_rows), num_columns)]
    vgs_rows_2d = [vgs_rows[i:i + num_columns] for i in range(0, len(vgs_rows), num_columns)]
    with open(output_csv, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(headers)
        if vth_rows_2d:
            csv_writer.writerows(vth_rows_2d)
        if vgs_rows_2d:
            csv_writer.writerows(vgs_rows_2d)
    return True

def format_csv_to_key_value(input_csv, output_txt):
    try:
        with open(input_csv, "r") as infile:
            csv_reader = csv.reader(infile)
            rows = list(csv_reader)
            if len(rows) < 3:
                with open(output_txt, "w") as outfile:
                    outfile.write("Vgs/Vth check not available (missing rows).")
                return
            headers, vth_values, vgs_values = rows[:3]  # Read first 3 rows

        filtered_lines = [
            f"vgs - vth value of {header}: {diff:.4f}"
            for header, vth, vgs in zip(headers, vth_values, vgs_values)
            if (diff := float(vgs) - float(vth)) < 0
        ]

        with open(output_txt, "w") as outfile:
            outfile.write("\n".join(filtered_lines) if filtered_lines else "No values found where vgs - vth < 0.")

        print("Filtered output written to:", output_txt)

    except Exception as e:
        print(f"An error occurred: {e}")

def read_txt_as_string(file_path):

    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def run_ngspice(circuit, filename):
    output_file = 'output/op.txt'
    cir_path = f'output/{filename}.cir'
    out_dir = os.path.dirname(cir_path)
    if not os.path.exists(out_dir):
        print(f'Path `{out_dir}` does not exist, creating it.')
        os.makedirs(out_dir, exist_ok=True)
    normalized = normalize_spice_includes(circuit)
    with open(cir_path, 'w') as f:
        f.write(normalized)

    # Locate ngspice executable (robust search):
    # 1. Respect environment variable NGSPICE_PATH
    # 2. If in conda, check $CONDA_PREFIX/bin/ngspice
    # 3. Try to find it in PATH via shutil.which
    # 4. Fallback to a list of common locations
    ngspice_path = os.getenv('NGSPICE_PATH')
    if not ngspice_path:
        conda_prefix = os.getenv('CONDA_PREFIX') or os.getenv('CONDA_DEFAULT_ENV')
        if conda_prefix:
            # If CONDA_PREFIX points to env root, prefer its bin
            candidate = os.path.join(os.getenv('CONDA_PREFIX', conda_prefix), 'bin', 'ngspice')
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                ngspice_path = candidate
    if not ngspice_path:
        ngspice_path = shutil.which('ngspice')
    if not ngspice_path:
        common_paths = [
            '/opt/homebrew/bin/ngspice',
            '/usr/local/bin/ngspice',
            '/usr/bin/ngspice',
            '/opt/local/bin/ngspice',
            '/home/chang/Documents/test/ngspice-44.2/release/src/ngspice'
        ]
        for p in common_paths:
            if os.path.exists(p) and os.access(p, os.X_OK):
                ngspice_path = p
                break

    # If ngspice is still not found, don't raise — write a clear placeholder and return False
    if not ngspice_path:
        msg = 'NGSPICE_NOT_FOUND: set NGSPICE_PATH or install ngspice (or place binary in conda env bin)'
        print(msg)
        # Write a minimal op.txt so downstream code can detect missing simulation
        with open(output_file, 'w') as f:
            f.write(msg + "\n")
        return False

    # Run ngspice with the discovered binary
    try:
        result = subprocess.run([ngspice_path, '-b', f'output/{filename}.cir'], capture_output=True, text=True)
        ngspice_output = result.stdout + ('\n' + result.stderr if result.stderr else '')
        with open(output_file, "w") as f:
            f.write(ngspice_output)
        if result.returncode != 0:
            print(f"NGspice failed with return code {result.returncode}")
            return False
        if "Could not find include file" in ngspice_output:
            print("NGspice include resolution failed; see output/op.txt")
            return False
    except Exception as e:
        ngspice_output = f"Error running NGspice: {str(e)}"
        with open(output_file, "w") as f:
            f.write(ngspice_output)
        print(ngspice_output)
        return False

    print("NGspice output written to", output_file)
    return True

"""
LLM can use this to decide which tool to call based on user input
here we have: dc_simulation, ac_simulation, transient_simulation, run_ngspice, ac_gain, tran_gain, output_swing, offset, icmr, bandwidth, unity_bandwidth, phase_margin, cmrr_tran, power, thd_input_range

"""
def tool_calling(tool_chain):
    global Gain_init, Bw_init, Pm_init, Dc_Gain_init,Tran_Gain_init, CMRR_init, Power_init, InputRange_Init,Thd_init, OW_init, Offset_init, UBw_init, ICMR_init
    gain = None
    Dc_gain = None
    tr_gain = None
    ow = None
    Offset = None
    bw = None
    ubw = None
    pm = None
    cmrr = None
    pr = None
    ir = None
    thd = None
    icmr = None
    sim_netlist = netlist

    input_txt = "output/op.txt"   # Replace with your actual input file
    filtered_txt = "output/vgscheck.txt"
    output_csv = "output/vgscheck.csv"
    output_txt = "output/vgscheck_output.txt"
    
    for tool_call in tool_chain['tool_calls']:
        if tool_call['name'].lower() == "dc_simulation":
            sim_netlist = dc_simulation(sim_netlist, source_names, output_nodes)

        elif tool_call['name'].lower() == "ac_simulation":
            sim_netlist = ac_simulation(sim_netlist, source_names, output_nodes)
            print(f"ac_netlist:{sim_netlist}")

        elif tool_call['name'].lower() == "transient_simulation":
            sim_netlist = trans_simulation(sim_netlist, source_names, output_nodes)

        elif tool_call['name'].lower() == "run_ngspice":
            ok = run_ngspice(sim_netlist, 'netlist')
            if ok:
                filter_lines(input_txt, filtered_txt)
                if convert_to_csv(filtered_txt, output_csv):
                    format_csv_to_key_value(output_csv, output_txt)
                    vgscheck = read_txt_as_string(output_txt)
                else:
                    vgscheck = "Vgs/Vth check not available (no parsed devices)."
            else:
                vgscheck = "NGspice run failed; Vgs/Vth check not available."
            #print(vgscheck)

        elif tool_call['name'].lower() == "ac_gain":   
            gain = ac_gain('output_ac')
            Gain_init = gain
            print(f"ac_gain result: {gain}")   

        elif tool_call['name'].lower() == "output_swing":   
            ow = out_swing('output_dc')
            OW_init = ow
            print(f"output swing result: {ow}") 
        
        elif tool_call['name'].lower() == "icmr":   
            icmr = ICMR('output_dc')
            ICMR_init = icmr
            print(f"input common mode voltage result: {icmr}")
        
        elif tool_call['name'].lower() == "offset":   
            Offset = offset('output_dc')
            Offset_init = Offset
            print(f"input offset result: {Offset}")

        elif tool_call['name'].lower() == "tran_gain":   
            tr_gain = tran_gain('output_tran')
            Tran_Gain_init = tr_gain
            print(f"tran_gain result: {tr_gain}")

        elif tool_call['name'].lower() == "bandwidth":
            bw = bandwidth('output_ac')
            Bw_init = bw
            print(f"bandwidth result: {bw}")

        elif tool_call['name'].lower() == "unity_bandwidth":
            ubw = unity_bandwidth('output_ac')
            UBw_init = ubw
            print(f"unity bandwidth result: {ubw}")

        elif tool_call['name'].lower() == "phase_margin":
            pm = phase_margin('output_ac')
            Pm_init = pm
            print(f"phase margin: {pm}")

        elif tool_call['name'].lower() == "cmrr_tran":
            cmrr,cmrr_max = cmrr_tran(sim_netlist)
            CMRR_init = cmrr
            print(f"cmrr: {cmrr}, cmrr_max: {cmrr_max}")

        elif tool_call['name'].lower() == "power":
            pr = stat_power('output_tran')
            Power_init = pr
            print(f"power: {pr}")
        
        elif tool_call['name'].lower() == "thd_input_range":
            thd, ir= thd_input_range('output_tran')
            Thd_init = thd
            InputRange_Init = ir
            print(f"thd is {thd}")

    sim_output = f"Transistors below vth: {vgscheck}," +  f"ac_gain is {gain}, " + f"tran_gain is {tr_gain}, " +  f"output swing is {ow}, " +  f"input offset is {Offset}, " +  f"input common mode voltage range is {icmr}, " + f"unity bandwidth is {ubw}, " + f"phase margin is {pm}, " + f"power is {pr}, " + f"cmrr is {cmrr},cmrr_max is {cmrr_max}," + f"thd is {thd},"
    
    return sim_output, sim_netlist

"""
负责把
"""
def extract_number(value):
    # Convert the value to a string (in case it's an integer or None)
    value_str = str(value) if value is not None else "0"
    
    # Use regex to find all numeric values (including decimals)
    match = re.search(r"[-+]?\d*\.\d+|\d+", value_str)
    if match:
        return float(match.group(0))
    return None
#%%
print(sim_question)
#%%
print(sizing_question)
#%%
sim_prompts_gen = prompts.build_simulation_prompt(sizing_question)
print(sim_prompts_gen)
#sim_prompts = make_chat_completion_request(sim_prompts_gen)
#%%
tool = make_chat_completion_request_function(sizing_question)
#%%
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
    max_iterations = 25
    tolerance = 0.05  # 5% tolerance
    iteration = 0
    converged = False

    gain_output = None
    tr_gain_output = None
    dc_gain_output = None
    bw_output = None
    ubw_output = None
    ow_output = None
    pm_output = None
    cmrr_output = None
    pr_output = None
    thd_output = None
    offset_output = None
    icmr_output = None

    input_txt = "output/op.txt"   # Replace with your actual input file
    filtered_txt = "output/vgscheck.txt"
    output_csv = "output/vgscheck.csv"
    output_txt = "output/vgscheck_output.txt"

    opti_output = None
    opti_netlist = sim_netlist
    previous_results_list =[]
    previous_results = [f"{sim_output}, " + f",the netlist is {opti_netlist}"]
    previous_results_list.append(f"{sim_output}, " + f",the netlist is {opti_netlist}")

    os.makedirs("output/90nm", exist_ok=True)

    with open("output/90nm/result_history.txt", 'w') as f:
        pass   # file is now empty

    with open("output/90nm/result_history.txt", 'w') as f:
            f.write(str(previous_results))

    gain_output_list = []
    dc_gain_output_list = []
    tr_gain_output_list = []
    bw_output_list = []
    ubw_output_list = []
    ow_output_list = []
    pm_output_list = []
    pr_output_list = []
    cmrr_output_list = []
    thd_output_list = []
    offset_output_list = []
    icmr_output_list = []
    
    target_values = target_values.strip().strip('`')  # Remove leading/trailing backticks and whitespace
    target_values = target_values.replace("json", "").strip()  # Remove the word "json" and any extra whitespace

    # Step 2: Parse the JSON string into a dictionary
    target_output = json.loads(target_values)
    for dict in target_output["target_values"]:
        for key, value in dict.items():
            globals()[key] = value
    
    gain_target = extracting_method(globals().get('ac_gain_target', '0')) if 'ac_gain_target' in globals() else None
    bandwidth_target = extracting_method(globals().get('bandwidth_target', '0')) if 'bandwidth_target' in globals() else None
    unity_bandwidth_target = extracting_method(globals().get('unity_bandwidth_target', '0')) if 'unity_bandwidth_target' in globals() else None
    phase_margin_target = extracting_method(globals().get('phase_margin_target', '0')) if 'phase_margin_target' in globals() else None
    tr_gain_target = extracting_method(globals().get('transient_gain_target', '0')) if 'transient_gain_target' in globals() else None
    input_offset_target = extracting_method(globals().get('input_offset_target', '0')) if 'input_offset_target' in globals() else None
    output_swing_target = extracting_method(globals().get('output_swing_target', '0')) if 'output_swing_target' in globals() else None
    pr_target = extracting_method(globals().get('power_target', '0')) if 'power_target' in globals() else None
    cmrr_target = extracting_method(globals().get('cmrr_target', '0')) if 'cmrr_target' in globals() else None
    thd_target = -np.abs(extracting_method(globals().get('thd_target', '0')) if 'thd_target' in globals() else None)
    icmr_target = extracting_method(globals().get('input_common_mode_range_target', '0')) if 'input_common_mode_range_target' in globals() else None


    gain_pass = True if gain_target not in globals() or gain_target is None else False
    tr_gain_pass = True if tr_gain_target not in globals() or tr_gain_target is None else False
    dc_gain_pass = True if tr_gain_target not in globals() or tr_gain_target is None else False
    ow_pass = True if output_swing_target not in globals() or output_swing_target is None else False
    bw_pass = True if bandwidth_target not in globals() or bandwidth_target is None else False
    ubw_pass = True if unity_bandwidth_target not in globals() or unity_bandwidth_target is None else False
    pm_pass = True if phase_margin_target not in globals() or phase_margin_target is None else False
    pr_pass = True if pr_target not in globals() or pr_target is None else False
    cmrr_pass = True if cmrr_target not in globals() or cmrr_target is None else False
    thd_pass = True if thd_target not in globals() or thd_target is None else False
    input_offset_pass = True if input_offset_target not in globals() or input_offset_target is None else False
    icmr_pass = True if icmr_target not in globals() or icmr_target is None else False

    sizing_Question = f"Currently, {sim_output}. " + sizing_question

    

    while iteration < max_iterations and not converged:
        time.sleep(20)
        with open("output/90nm/result_history.txt", 'r') as f:
            previous_results = f.read()
        print(f"----------------------iter = {iteration}-----------------------------")
        #print(f"previous_results:{previous_results}")
        #######################observation#########################
        analysising_prompt = prompts.build_analysis_prompt(previous_results, sizing_question)
        analysis = make_chat_completion_request(analysising_prompt)
        print(analysis)
        #analysis_content = analysis["text"]
        ##########################analysis#######################
        time.sleep(10)
        optimising_prompt = prompts.build_optimising_prompt(type_identified, analysis, previous_results)
        optimising = make_chat_completion_request(optimising_prompt)
        print(optimising)
        ###########################sizing/optimising######################
        time.sleep(10)
        sizing_prompt = prompts.build_sizing_prompt(sizing_Question, opti_netlist, optimising)
        modified = make_chat_completion_request(sizing_prompt)
        print(modified)
        modified_output = modified
        print("----------------------Modified-----------------------------")
        print(modified_output)
        time.sleep(10)
        ###########################function calling######################
        print("------------------------result-----------------------------")
        for tool_call in tools['tool_calls']:
            opti_netlist = extract_code(modified_output)
            print(opti_netlist)

            if tool_call['name'].lower() == "run_ngspice":
                run_ngspice(opti_netlist, 'netlist')
                print("running ngspice")
                filter_lines(input_txt, filtered_txt)
                convert_to_csv(filtered_txt, output_csv)
                format_csv_to_key_value(output_csv, output_txt)
                vgscheck = read_txt_as_string(output_txt)
        
            elif tool_call['name'].lower() == "ac_gain":
                #run_ngspice(sim_netlist)
                gain_output = ac_gain('output_ac')
                #print(f"ac_gain result: {gain_output}") 

            elif tool_call['name'].lower() == "dc_gain":
                #run_ngspice(sim_netlist)
                dc_gain_output = dc_gain('output_dc')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "output_swing":
                #run_ngspice(sim_netlist)
                ow_output = out_swing('output_dc')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "offset":
                #run_ngspice(sim_netlist)
                offset_output = offset('output_dc')

            elif tool_call['name'].lower() == "icmr":
                #run_ngspice(sim_netlist)
                icmr_output = ICMR('output_dc')

            elif tool_call['name'].lower() == "tran_gain":
                #run_ngspice(sim_netlist)
                tr_gain_output = tran_gain('output_tran')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "bandwidth":
                #run_ngspice(sim_netlist)
                bw_output = bandwidth('output_ac')
                #print(f"bandwidth result: {bw_output}")

            elif tool_call['name'].lower() == "unity_bandwidth":
                #run_ngspice(sim_netlist)
                ubw_output = unity_bandwidth('output_ac')
                #print(f"bandwidth result: {bw_output}")

            elif tool_call['name'].lower() == "phase_margin":
                #run_ngspice(sim_netlist)
                pm_output = phase_margin('output_ac')
                #print(f"phase margin result: {pm_output}")

            elif tool_call['name'].lower() == "power":
                #run_ngspice(sim_netlist)
                pr_output = stat_power('output_tran')
                #print(f"phase margin result: {pm_output}")

            elif tool_call['name'].lower() == "thd_input_range":
                #run_ngspice(sim_netlist)
                thd_output, ir_output = thd_input_range('output_tran')

            elif tool_call['name'].lower() == "cmrr_tran":
                #run_ngspice(sim_netlist)
                cmrr_output,cmrr_max = cmrr_tran(opti_netlist)

            
        
        #print(vgscheck)
            
        opti_output = f"Transistors below vth: {vgscheck}," + f"ac_gain is {gain_output} dB, " + f"tran_gain is {tr_gain_output} dB, " + f"output_swing is {ow_output}, " + f"input offset is {offset_output}, " + f"input common mode voltage range is {icmr_output}, "  + f"unity bandwidth is {ubw_output}, " + f"phase margin is {pm_output}, " + f"power is {pr_output}, " + f"cmrr is {cmrr_output} cmrr max is {cmrr_max}," + f"thd is {thd_output}," 

        #save the output value in a list
        gain_output_list.append(gain_output)
        tr_gain_output_list.append(tr_gain_output)
        dc_gain_output_list.append(dc_gain_output)
        ow_output_list.append(ow_output)
        bw_output_list.append(bw_output)
        ubw_output_list.append(ubw_output)
        pm_output_list.append(pm_output)
        pr_output_list.append(pr_output)
        cmrr_output_list.append(cmrr_output)
        thd_output_list.append(thd_output)
        offset_output_list.append(offset_output)
        icmr_output_list.append(icmr_output)
                    
        print(opti_output)

        #comparison
        if gain_target is not None:
            if gain_output >= gain_target - gain_target * tolerance:
                gain_pass = True
            else:
                gain_pass = False

        if tr_gain_target is not None:
            if tr_gain_output >= tr_gain_target - tr_gain_target * tolerance:
                tr_gain_pass = True
            else:
                tr_gain_pass = False

        if output_swing_target is not None:
            if ow_output >= output_swing_target - output_swing_target * tolerance:
                ow_pass = True
            else:
                ow_pass = False

        if input_offset_target is not None:
            if offset_output <= input_offset_target - input_offset_target * tolerance:
                input_offset_pass = True
            else:
                input_offset_pass = False     

        if icmr_target is not None:
            if icmr_output >= icmr_target - icmr_target * tolerance:
                icmr_pass = True
            else:
                icmr_pass = False   
        
        if bandwidth_target is not None:
            if bw_output >= bandwidth_target - bandwidth_target * tolerance:
                bw_pass = True
            else:
                bw_pass = False

        if unity_bandwidth_target is not None:
            if ubw_output >= unity_bandwidth_target - unity_bandwidth_target * tolerance:
                ubw_pass = True
            else:
                ubw_pass = False

        if phase_margin_target is not None:
            if pm_output >= phase_margin_target - phase_margin_target * tolerance:
                pm_pass = True
            else:
                pm_pass = False

        if pr_target is not None:
            if pr_output <= pr_target + pr_target * tolerance:
                pr_pass = True
            else:
                pr_pass = False

        if cmrr_target is not None:
            if cmrr_output >= cmrr_target - cmrr_target * tolerance:
                cmrr_pass = True
            else:
                cmrr_pass = False

        if thd_target is not None:
            if thd_output <= thd_target + np.abs(thd_target) * tolerance:
                thd_pass = True
            else:
                thd_pass = False


        if gain_pass and ubw_pass and pm_pass and tr_gain_pass and pr_pass and cmrr_pass and dc_gain_pass and thd_pass and ow_pass and input_offset_pass and icmr_pass and vgscheck == "No values found where vgs - vth < 0.":
            converged = True

        sizing_Question = f"Currently,{opti_output}" + sizing_question
        pass_or_not = f"gain_pass:{gain_pass},tr_gain_pass:{tr_gain_pass},output_swing_pass:{ow_pass},input_offset_pass:{input_offset_pass}, icmr_pass:{icmr_pass}, unity_bandwidth_pass:{ubw_pass}, phase_margin_pass:{pm_pass}, power_pass:{pr_pass}, cmrr_pass:{cmrr_pass} , thd_pass:{thd_pass}"
        iteration += 1
        previous_results_list.append(f"Currently, {opti_output}, {pass_or_not},the netlist is {opti_netlist}")
        if len(previous_results_list) > 5:
            previous_results_list.pop(0)  # Remove the oldest result
        
        #store the his tory in a file
        with open("output/90nm/result_history.txt", 'w') as f:
            f.write(str(previous_results_list))

        print(f"gain_target:{gain_target}, tr_gain_target:{tr_gain_target},output_swing_target:{output_swing_target}, input_offset_target:{input_offset_target}, icmr_target:{icmr_target}, unity_bandwidth_target:{unity_bandwidth_target}, phase_margin_target:{phase_margin_target}, power_target:{pr_target}, cmrr_target:{cmrr_target}, thd_target:{thd_target}")
        print(f"gain_pass:{gain_pass},tr_gain_pass:{tr_gain_pass},output_swing_pass:{ow_pass},input_offset_pass:{input_offset_pass}, icmr_pass:{icmr_pass}, unity_bandwidth_pass:{ubw_pass}, phase_margin_pass:{pm_pass}, power_pass:{pr_pass}, cmrr_pass:{cmrr_pass} , thd_pass:{thd_pass}")
    ##########################################################################################################
    # save the value in file
    file_empty = not os.path.exists('output/90nm/g2_o3.csv') or os.stat('output/90nm/g2_o3.csv').st_size == 0
    with open('output/90nm/g2_o3.csv', 'a', newline='') as csvfile:
        fieldnames = ['iteration', 'gain_output', 'tr_gain_output', 'output_swing_output', 'input_offset_output',  'icmr_output', 'ubw_output', 'pm_output', 'pr_output', 'cmrr_output', 'thd_output' ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()
        writer.writerow({'iteration': 0, 'gain_output': Gain_init,'tr_gain_output': Tran_Gain_init, 'output_swing_output': OW_init, 'input_offset_output': Offset_init, 'icmr_output': ICMR_init, 'ubw_output': UBw_init, 'pm_output': Pm_init, 'pr_output':Power_init, 'cmrr_output': CMRR_init, 'thd_output': Thd_init})
        for i in range(len(gain_output_list)):
            writer.writerow({'iteration': i+1, 'gain_output': gain_output_list[i], 'tr_gain_output': tr_gain_output_list[i], 'output_swing_output': ow_output_list[i], 'input_offset_output': offset_output_list[i], 'icmr_output': icmr_output_list[i], 'ubw_output': ubw_output_list[i], 'pm_output': pm_output_list[i], 'pr_output': pr_output_list[i], 'cmrr_output': cmrr_output_list[i], 'thd_output': thd_output_list[i] })

    return {'converged': converged, 
            'iterations': iteration, 
            'gain_output': gain_output if 'gain_output' in locals() else None, 
            'tr_gain_output': tr_gain_output if 'tr_gain_output' in locals() else None,
            'output_swing_output': ow_output if 'ow_output' in locals() else None,
            'input_offset_output': offset_output if 'offset_output' in locals() else None,
            'ubw_output': ubw_output if 'ubw_output' in locals() else None,
            'pm_output':pm_output if 'pm_output' in locals() else None,
            'pr_output':pr_output if 'pr_output' in locals() else None,
            'cmrr_output':cmrr_output if 'cmrr_output' in locals() else None,
            'icmr_output':icmr_output if 'icmr_output' in locals() else None,
            'thd_output':thd_output if 'thd_output' in locals() else None}, opti_netlist
#%%
def run_multiple_optimizations(target_values, sim_netlist, num_runs=1):
    results = []  # List to store results of each run

    for i in range(num_runs):
        print(f"Starting optimization run {i + 1}")
        result, opti_netlist = optimization(tool_chain, target_values, sim_netlist, extract_number)
        results.append(result)  # Append the result of each run to the results list
        print(f"Run {i + 1} result: {result}")
        print("----------------------\n")    
results = run_multiple_optimizations(target_values, sim_netlist)
#%%
# save the netlist
#print("Summary of all optimization runs:")
source_file = 'output/netlist.cir'
destination_file = f'output/90nm/netlist_cs_o3/a1.cir'
shutil.copyfile(source_file, destination_file)
print(f"Netlist copy to {destination_file}")
#%%
df = pd.read_csv('output/90nm/g2_o3.csv')
df['batch'] = (df['iteration'] == 0).cumsum()
df['bw_output_dB'] = 20 * np.log10(df['ubw_output'] + 1e-9)

# Filter the DataFrame for the first 5 batches
filtered_df = df[df['batch'] <= 5]

# Create a 4x2 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Adjust figure size for 4x2 layout
fig.subplots_adjust(hspace=0.1, wspace=0.3)  # Adjust spacing between subplots

# Generate a list of colors
colors = plt.cm.viridis(np.linspace(0, 0.9, filtered_df['batch'].max() + 1))
# Function to plot in a subplot
def plot_subplot(ax, x, y, xlabel, ylabel, ylim_min, ylim_max, fill_range=None, fill_label=None, log_scale=False):
    for batch, group in filtered_df.groupby('batch'):
        ax.plot(group[x], group[y], marker='o', color=colors[batch], label=f'Attempt {batch}', markersize=4, linewidth=1)
    ax.set_xlim(filtered_df[x].min() - 1, filtered_df[x].max() + 1)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    #ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(False)
    if fill_range:
        ax.fill_between(ax.get_xlim(), fill_range[0], fill_range[1], color='blue', alpha=0.1, label=fill_label)
    if log_scale:
        ax.set_yscale('log')
    #ax.legend(fontsize=10)

# Plot each metric in a subplot
#plot_subplot(axes[0, 0], 'iteration', 'gain_output', 'Iterations', 'Gain (dB)', 'Gain', 0, 120, fill_range=(65, 85), fill_label='Target Range')
plot_subplot(axes[0, 1], 'iteration', 'pm_output', 'Iterations', 'Phase Margin (°)', -5, 115, fill_range=(50, 150), fill_label='Target Range')
plot_subplot(axes[0, 2], 'iteration', 'bw_output_dB', 'Iterations', 'UGBW (dB)', 90, 20 * np.log10(1e7) + 35, fill_range=(20 * np.log10(1e7), 20 * np.log10(1e7) + 35), fill_label='Target Range')
plot_subplot(axes[1, 1], 'iteration', 'tr_gain_output', 'Iterations', 'Gain (dB)', 15, 85, fill_range=(61.75, 90), fill_label='Target Range')
plot_subplot(axes[2, 1], 'iteration', 'pr_output', 'Iterations', 'Power (W)', -0.0005, 0.015, fill_range=(-0.0005, 0.0105), fill_label='Target Range')
plot_subplot(axes[1, 2], 'iteration', 'cmrr_output', 'Iterations', 'CMRR (dB)', 0, 130, fill_range=(100 * 0.95, 160), fill_label='Target Range')
plot_subplot(axes[1, 0], 'iteration', 'output_swing_output', 'Iterations', 'Output Swing (V)', 0, 1.3, fill_range=(1.2 * 0.95, 1.3), fill_label='Target Range')
plot_subplot(axes[2, 2], 'iteration', 'thd_output', 'Iterations', 'THD (dB)', -50, 0, fill_range=(-60, -24.7), fill_label='Target Range')
plot_subplot(axes[2, 0], 'iteration', 'input_offset_output', 'Iterations', 'Offset (V)', -0.05, 0.2, fill_range=(-0.1, 0.001), fill_label='Target Range')
plot_subplot(axes[0, 0], 'iteration', 'icmr_output', 'Iterations', 'Input Range', 0, 1.3, fill_range=(1.2 * 0.95, 1.3), fill_label='Target Range')

labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
for i, ax in enumerate(axes.flat):
    ax.text(1, 0.1, labels[i], transform=ax.transAxes, fontsize=14, va='top', ha='right')

for ax in axes[:-1, :].flat:
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_xlabel('')  # Remove x-axis label

# Create a common legend
handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=6, fontsize=13)

# Save the figure
plt.savefig('railtorail_subplots_4x2_g1.pdf', format='pdf', bbox_inches='tight')
plt.show()
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
