"""Simulation helpers and metric calculations."""

import csv
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from agent_test_gpt.netlist_utils import normalize_spice_includes


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

    # If ngspice is still not found, don't raise - write a clear placeholder and return False
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
