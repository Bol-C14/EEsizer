"""Tool calling and optimization runners."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import csv
import json
import os
import time

import numpy as np

from agent_test_gpt import prompts
from agent_test_gpt import config
from agent_test_gpt.simulation_utils import _ensure_flat_str_list


@dataclass
class ToolCallingContext:
    netlist: str
    source_names: List[str]
    output_nodes: List[str]


@dataclass
class ToolCallingConfig:
    input_txt: str = config.OP_TXT_PATH
    filtered_txt: str = config.VGS_FILTERED_PATH
    output_csv: str = config.VGS_CSV_PATH
    output_txt: str = config.VGS_OUTPUT_PATH


@dataclass
class ToolCallingDeps:
    dc_simulation: Callable
    ac_simulation: Callable
    trans_simulation: Callable
    run_ngspice: Callable
    filter_lines: Callable
    convert_to_csv: Callable
    format_csv_to_key_value: Callable
    read_txt_as_string: Callable
    ac_gain: Callable
    out_swing: Callable
    ICMR: Callable
    offset: Callable
    tran_gain: Callable
    bandwidth: Callable
    unity_bandwidth: Callable
    phase_margin: Callable
    cmrr_tran: Callable
    stat_power: Callable
    thd_input_range: Callable


@dataclass
class ToolCallingResult:
    sim_output: str
    sim_netlist: str
    metrics: Dict[str, float | None]
    vgscheck: str | None
    cmrr_max: float | None


class ToolChainRunner:
    def __init__(self, context: ToolCallingContext, deps: ToolCallingDeps, config: ToolCallingConfig | None = None):
        self.context = context
        self.deps = deps
        self.config = config or ToolCallingConfig()

    def _run_ngspice_with_vgscheck(self, sim_netlist: str) -> str:
        ok = self.deps.run_ngspice(sim_netlist, "netlist")
        if ok:
            self.deps.filter_lines(self.config.input_txt, self.config.filtered_txt)
            if self.deps.convert_to_csv(self.config.filtered_txt, self.config.output_csv):
                self.deps.format_csv_to_key_value(self.config.output_csv, self.config.output_txt)
                return self.deps.read_txt_as_string(self.config.output_txt)
            return "Vgs/Vth check not available (no parsed devices)."
        return "NGspice run failed; Vgs/Vth check not available."

    def run(self, tool_chain: Dict) -> ToolCallingResult:
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
        vgscheck = None
        cmrr_max = None
        sim_netlist = self.context.netlist

        source_names = _ensure_flat_str_list("context.source_names", self.context.source_names)
        output_nodes = _ensure_flat_str_list("context.output_nodes", self.context.output_nodes)

        for tool_call in tool_chain["tool_calls"]:
            tool_name = tool_call["name"].lower()
            if tool_name == "dc_simulation":
                sim_netlist = self.deps.dc_simulation(sim_netlist, source_names, output_nodes)
            elif tool_name == "ac_simulation":
                sim_netlist = self.deps.ac_simulation(sim_netlist, source_names, output_nodes)
                print(f"ac_netlist:{sim_netlist}")
            elif tool_name == "transient_simulation":
                sim_netlist = self.deps.trans_simulation(sim_netlist, source_names, output_nodes)
            elif tool_name == "run_ngspice":
                vgscheck = self._run_ngspice_with_vgscheck(sim_netlist)
            elif tool_name == "ac_gain":
                gain = self.deps.ac_gain("output_ac")
                print(f"ac_gain result: {gain}")
            elif tool_name == "output_swing":
                ow = self.deps.out_swing("output_dc")
                print(f"output swing result: {ow}")
            elif tool_name == "icmr":
                icmr = self.deps.ICMR("output_dc")
                print(f"input common mode voltage result: {icmr}")
            elif tool_name == "offset":
                Offset = self.deps.offset("output_dc")
                print(f"input offset result: {Offset}")
            elif tool_name == "tran_gain":
                tr_gain = self.deps.tran_gain("output_tran")
                print(f"tran_gain result: {tr_gain}")
            elif tool_name == "bandwidth":
                bw = self.deps.bandwidth("output_ac")
                print(f"bandwidth result: {bw}")
            elif tool_name == "unity_bandwidth":
                ubw = self.deps.unity_bandwidth("output_ac")
                print(f"unity bandwidth result: {ubw}")
            elif tool_name == "phase_margin":
                pm = self.deps.phase_margin("output_ac")
                print(f"phase margin: {pm}")
            elif tool_name == "cmrr_tran":
                cmrr, cmrr_max = self.deps.cmrr_tran(sim_netlist)
                print(f"cmrr: {cmrr}, cmrr_max: {cmrr_max}")
            elif tool_name == "power":
                pr = self.deps.stat_power("output_tran")
                print(f"power: {pr}")
            elif tool_name == "thd_input_range":
                thd, ir = self.deps.thd_input_range("output_tran")
                print(f"thd is {thd}")

        sim_output = (
            f"Transistors below vth: {vgscheck},"
            + f"ac_gain is {gain}, "
            + f"tran_gain is {tr_gain}, "
            + f"output swing is {ow}, "
            + f"input offset is {Offset}, "
            + f"input common mode voltage range is {icmr}, "
            + f"unity bandwidth is {ubw}, "
            + f"phase margin is {pm}, "
            + f"power is {pr}, "
            + f"cmrr is {cmrr},cmrr_max is {cmrr_max},"
            + f"thd is {thd},"
        )

        metrics = {
            "gain": gain,
            "dc_gain": Dc_gain,
            "tran_gain": tr_gain,
            "output_swing": ow,
            "offset": Offset,
            "bandwidth": bw,
            "unity_bandwidth": ubw,
            "phase_margin": pm,
            "cmrr": cmrr,
            "power": pr,
            "input_range": ir,
            "thd": thd,
            "icmr": icmr,
        }

        return ToolCallingResult(
            sim_output=sim_output,
            sim_netlist=sim_netlist,
            metrics=metrics,
            vgscheck=vgscheck,
            cmrr_max=cmrr_max,
        )


@dataclass
class OptimizationContext:
    sim_output: str
    sizing_question: str
    type_identified: str


@dataclass
class OptimizationConfig:
    max_iterations: int = config.MAX_ITERATIONS
    tolerance: float = config.TOLERANCE
    output_dir: str = config.OUTPUT_DIR
    history_file: str = config.RESULT_HISTORY_FILE
    csv_file: str = config.CSV_FILE
    input_txt: str = config.OP_TXT_PATH
    filtered_txt: str = config.VGS_FILTERED_PATH
    output_csv: str = config.VGS_CSV_PATH
    output_txt: str = config.VGS_OUTPUT_PATH


@dataclass
class OptimizationDeps:
    make_chat_completion_request: Callable
    run_ngspice: Callable
    filter_lines: Callable
    convert_to_csv: Callable
    format_csv_to_key_value: Callable
    read_txt_as_string: Callable
    ac_gain: Callable
    dc_gain: Callable
    out_swing: Callable
    offset: Callable
    ICMR: Callable
    tran_gain: Callable
    bandwidth: Callable
    unity_bandwidth: Callable
    phase_margin: Callable
    stat_power: Callable
    thd_input_range: Callable
    cmrr_tran: Callable
    extract_code: Callable


def _parse_target_values(target_values: str, extracting_method: Callable) -> Tuple[Dict[str, float | None], Dict[str, bool]]:
    target_values = target_values.strip().strip("`")
    target_values = target_values.replace("json", "").strip()

    target_output = json.loads(target_values)
    for target_dict in target_output["target_values"]:
        for key, value in target_dict.items():
            globals()[key] = value

    gain_target = extracting_method(globals().get("ac_gain_target", "0")) if "ac_gain_target" in globals() else None
    bandwidth_target = extracting_method(globals().get("bandwidth_target", "0")) if "bandwidth_target" in globals() else None
    unity_bandwidth_target = extracting_method(globals().get("unity_bandwidth_target", "0")) if "unity_bandwidth_target" in globals() else None
    phase_margin_target = extracting_method(globals().get("phase_margin_target", "0")) if "phase_margin_target" in globals() else None
    tr_gain_target = extracting_method(globals().get("transient_gain_target", "0")) if "transient_gain_target" in globals() else None
    input_offset_target = extracting_method(globals().get("input_offset_target", "0")) if "input_offset_target" in globals() else None
    output_swing_target = extracting_method(globals().get("output_swing_target", "0")) if "output_swing_target" in globals() else None
    pr_target = extracting_method(globals().get("power_target", "0")) if "power_target" in globals() else None
    cmrr_target = extracting_method(globals().get("cmrr_target", "0")) if "cmrr_target" in globals() else None
    thd_target = -np.abs(extracting_method(globals().get("thd_target", "0")) if "thd_target" in globals() else None)
    icmr_target = extracting_method(globals().get("input_common_mode_range_target", "0")) if "input_common_mode_range_target" in globals() else None

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

    targets = {
        "gain_target": gain_target,
        "bandwidth_target": bandwidth_target,
        "unity_bandwidth_target": unity_bandwidth_target,
        "phase_margin_target": phase_margin_target,
        "tr_gain_target": tr_gain_target,
        "input_offset_target": input_offset_target,
        "output_swing_target": output_swing_target,
        "pr_target": pr_target,
        "cmrr_target": cmrr_target,
        "thd_target": thd_target,
        "icmr_target": icmr_target,
    }
    passes = {
        "gain_pass": gain_pass,
        "tr_gain_pass": tr_gain_pass,
        "dc_gain_pass": dc_gain_pass,
        "ow_pass": ow_pass,
        "bw_pass": bw_pass,
        "ubw_pass": ubw_pass,
        "pm_pass": pm_pass,
        "pr_pass": pr_pass,
        "cmrr_pass": cmrr_pass,
        "thd_pass": thd_pass,
        "input_offset_pass": input_offset_pass,
        "icmr_pass": icmr_pass,
    }
    return targets, passes


def _write_history(path: str, items: List[str]) -> None:
    with open(path, "w") as f:
        f.write(str(items))


class OptimizationRunner:
    def __init__(self, context: OptimizationContext, deps: OptimizationDeps, config: OptimizationConfig | None = None):
        self.context = context
        self.deps = deps
        self.config = config or OptimizationConfig()

    def _initialize_history(self, sim_netlist: str) -> List[str]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        previous_results_list = []
        previous_results = [f"{self.context.sim_output}, " + f",the netlist is {sim_netlist}"]
        previous_results_list.append(f"{self.context.sim_output}, " + f",the netlist is {sim_netlist}")
        _write_history(self.config.history_file, previous_results)
        return previous_results_list

    def _run_llm_cycle(self, previous_results: str, sizing_question: str, opti_netlist: str) -> str:
        analysising_prompt = prompts.build_analysis_prompt(previous_results, self.context.sizing_question)
        analysis = self.deps.make_chat_completion_request(analysising_prompt)
        print(analysis)
        time.sleep(10)

        optimising_prompt = prompts.build_optimising_prompt(self.context.type_identified, analysis, previous_results)
        optimising = self.deps.make_chat_completion_request(optimising_prompt)
        print(optimising)
        time.sleep(10)

        sizing_prompt = prompts.build_sizing_prompt(sizing_question, opti_netlist, optimising)
        modified = self.deps.make_chat_completion_request(sizing_prompt)
        print(modified)
        return modified

    def _run_tool_calls(self, tools: Dict, modified_output: str) -> Tuple[Dict, str, str, float | None]:
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
        vgscheck = None
        cmrr_max = None
        opti_netlist = None

        for tool_call in tools["tool_calls"]:
            opti_netlist = self.deps.extract_code(modified_output)
            print(opti_netlist)
            tool_name = tool_call["name"].lower()

            if tool_name == "run_ngspice":
                self.deps.run_ngspice(opti_netlist, "netlist")
                print("running ngspice")
                self.deps.filter_lines(self.config.input_txt, self.config.filtered_txt)
                self.deps.convert_to_csv(self.config.filtered_txt, self.config.output_csv)
                self.deps.format_csv_to_key_value(self.config.output_csv, self.config.output_txt)
                vgscheck = self.deps.read_txt_as_string(self.config.output_txt)
            elif tool_name == "ac_gain":
                gain_output = self.deps.ac_gain("output_ac")
            elif tool_name == "dc_gain":
                dc_gain_output = self.deps.dc_gain("output_dc")
            elif tool_name == "output_swing":
                ow_output = self.deps.out_swing("output_dc")
            elif tool_name == "offset":
                offset_output = self.deps.offset("output_dc")
            elif tool_name == "icmr":
                icmr_output = self.deps.ICMR("output_dc")
            elif tool_name == "tran_gain":
                tr_gain_output = self.deps.tran_gain("output_tran")
            elif tool_name == "bandwidth":
                bw_output = self.deps.bandwidth("output_ac")
            elif tool_name == "unity_bandwidth":
                ubw_output = self.deps.unity_bandwidth("output_ac")
            elif tool_name == "phase_margin":
                pm_output = self.deps.phase_margin("output_ac")
            elif tool_name == "power":
                pr_output = self.deps.stat_power("output_tran")
            elif tool_name == "thd_input_range":
                thd_output, _ir_output = self.deps.thd_input_range("output_tran")
            elif tool_name == "cmrr_tran":
                cmrr_output, cmrr_max = self.deps.cmrr_tran(opti_netlist)

        metrics = {
            "gain_output": gain_output,
            "tr_gain_output": tr_gain_output,
            "dc_gain_output": dc_gain_output,
            "bw_output": bw_output,
            "ubw_output": ubw_output,
            "ow_output": ow_output,
            "pm_output": pm_output,
            "cmrr_output": cmrr_output,
            "pr_output": pr_output,
            "thd_output": thd_output,
            "offset_output": offset_output,
            "icmr_output": icmr_output,
        }
        return metrics, vgscheck, opti_netlist, cmrr_max

    def _append_metrics(self, lists: Dict[str, List], metrics: Dict[str, float | None]) -> None:
        lists["gain_output_list"].append(metrics["gain_output"])
        lists["tr_gain_output_list"].append(metrics["tr_gain_output"])
        lists["dc_gain_output_list"].append(metrics["dc_gain_output"])
        lists["ow_output_list"].append(metrics["ow_output"])
        lists["bw_output_list"].append(metrics["bw_output"])
        lists["ubw_output_list"].append(metrics["ubw_output"])
        lists["pm_output_list"].append(metrics["pm_output"])
        lists["pr_output_list"].append(metrics["pr_output"])
        lists["cmrr_output_list"].append(metrics["cmrr_output"])
        lists["thd_output_list"].append(metrics["thd_output"])
        lists["offset_output_list"].append(metrics["offset_output"])
        lists["icmr_output_list"].append(metrics["icmr_output"])

    def _update_pass_flags(self, targets: Dict[str, float | None], metrics: Dict[str, float | None], passes: Dict[str, bool]) -> None:
        tol = self.config.tolerance

        if targets["gain_target"] is not None:
            passes["gain_pass"] = metrics["gain_output"] >= targets["gain_target"] - targets["gain_target"] * tol
        if targets["tr_gain_target"] is not None:
            passes["tr_gain_pass"] = metrics["tr_gain_output"] >= targets["tr_gain_target"] - targets["tr_gain_target"] * tol
        if targets["output_swing_target"] is not None:
            passes["ow_pass"] = metrics["ow_output"] >= targets["output_swing_target"] - targets["output_swing_target"] * tol
        if targets["input_offset_target"] is not None:
            passes["input_offset_pass"] = metrics["offset_output"] <= targets["input_offset_target"] - targets["input_offset_target"] * tol
        if targets["icmr_target"] is not None:
            passes["icmr_pass"] = metrics["icmr_output"] >= targets["icmr_target"] - targets["icmr_target"] * tol
        if targets["bandwidth_target"] is not None:
            passes["bw_pass"] = metrics["bw_output"] >= targets["bandwidth_target"] - targets["bandwidth_target"] * tol
        if targets["unity_bandwidth_target"] is not None:
            passes["ubw_pass"] = metrics["ubw_output"] >= targets["unity_bandwidth_target"] - targets["unity_bandwidth_target"] * tol
        if targets["phase_margin_target"] is not None:
            passes["pm_pass"] = metrics["pm_output"] >= targets["phase_margin_target"] - targets["phase_margin_target"] * tol
        if targets["pr_target"] is not None:
            passes["pr_pass"] = metrics["pr_output"] <= targets["pr_target"] + targets["pr_target"] * tol
        if targets["cmrr_target"] is not None:
            passes["cmrr_pass"] = metrics["cmrr_output"] >= targets["cmrr_target"] - targets["cmrr_target"] * tol
        if targets["thd_target"] is not None:
            passes["thd_pass"] = metrics["thd_output"] <= targets["thd_target"] + np.abs(targets["thd_target"]) * tol

    def _write_csv_results(self, initial_metrics: Dict[str, float | None], lists: Dict[str, List]) -> None:
        file_empty = not os.path.exists(self.config.csv_file) or os.stat(self.config.csv_file).st_size == 0
        with open(self.config.csv_file, "a", newline="") as csvfile:
            fieldnames = [
                "iteration",
                "gain_output",
                "tr_gain_output",
                "output_swing_output",
                "input_offset_output",
                "icmr_output",
                "ubw_output",
                "pm_output",
                "pr_output",
                "cmrr_output",
                "thd_output",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if file_empty:
                writer.writeheader()
            writer.writerow(
                {
                    "iteration": 0,
                    "gain_output": initial_metrics.get("gain_output"),
                    "tr_gain_output": initial_metrics.get("tr_gain_output"),
                    "output_swing_output": initial_metrics.get("output_swing_output"),
                    "input_offset_output": initial_metrics.get("input_offset_output"),
                    "icmr_output": initial_metrics.get("icmr_output"),
                    "ubw_output": initial_metrics.get("ubw_output"),
                    "pm_output": initial_metrics.get("pm_output"),
                    "pr_output": initial_metrics.get("pr_output"),
                    "cmrr_output": initial_metrics.get("cmrr_output"),
                    "thd_output": initial_metrics.get("thd_output"),
                }
            )
            for i in range(len(lists["gain_output_list"])):
                writer.writerow(
                    {
                        "iteration": i + 1,
                        "gain_output": lists["gain_output_list"][i],
                        "tr_gain_output": lists["tr_gain_output_list"][i],
                        "output_swing_output": lists["ow_output_list"][i],
                        "input_offset_output": lists["offset_output_list"][i],
                        "icmr_output": lists["icmr_output_list"][i],
                        "ubw_output": lists["ubw_output_list"][i],
                        "pm_output": lists["pm_output_list"][i],
                        "pr_output": lists["pr_output_list"][i],
                        "cmrr_output": lists["cmrr_output_list"][i],
                        "thd_output": lists["thd_output_list"][i],
                    }
                )

    def run(
        self,
        tools: Dict,
        target_values: str,
        sim_netlist: str,
        extracting_method: Callable,
        initial_metrics: Dict[str, float | None],
    ) -> Tuple[Dict, str]:
        max_iterations = self.config.max_iterations
        iteration = 0
        converged = False

        lists = {
            "gain_output_list": [],
            "dc_gain_output_list": [],
            "tr_gain_output_list": [],
            "bw_output_list": [],
            "ubw_output_list": [],
            "ow_output_list": [],
            "pm_output_list": [],
            "pr_output_list": [],
            "cmrr_output_list": [],
            "thd_output_list": [],
            "offset_output_list": [],
            "icmr_output_list": [],
        }

        targets, passes = _parse_target_values(target_values, extracting_method)
        opti_netlist = sim_netlist
        previous_results_list = self._initialize_history(opti_netlist)
        sizing_Question = f"Currently, {self.context.sim_output}. " + self.context.sizing_question

        while iteration < max_iterations and not converged:
            time.sleep(20)
            with open(self.config.history_file, "r") as f:
                previous_results = f.read()
            print(f"----------------------iter = {iteration}-----------------------------")

            modified_output = self._run_llm_cycle(previous_results, sizing_Question, opti_netlist)
            print("----------------------Modified-----------------------------")
            print(modified_output)
            time.sleep(10)

            print("------------------------result-----------------------------")
            metrics, vgscheck, opti_netlist, cmrr_max = self._run_tool_calls(tools, modified_output)

            opti_output = (
                f"Transistors below vth: {vgscheck},"
                + f"ac_gain is {metrics['gain_output']} dB, "
                + f"tran_gain is {metrics['tr_gain_output']} dB, "
                + f"output_swing is {metrics['ow_output']}, "
                + f"input offset is {metrics['offset_output']}, "
                + f"input common mode voltage range is {metrics['icmr_output']}, "
                + f"unity bandwidth is {metrics['ubw_output']}, "
                + f"phase margin is {metrics['pm_output']}, "
                + f"power is {metrics['pr_output']}, "
                + f"cmrr is {metrics['cmrr_output']} cmrr max is {cmrr_max},"
                + f"thd is {metrics['thd_output']},"
            )

            self._append_metrics(lists, metrics)
            print(opti_output)

            self._update_pass_flags(targets, metrics, passes)

            if (
                passes["gain_pass"]
                and passes["ubw_pass"]
                and passes["pm_pass"]
                and passes["tr_gain_pass"]
                and passes["pr_pass"]
                and passes["cmrr_pass"]
                and passes["dc_gain_pass"]
                and passes["thd_pass"]
                and passes["ow_pass"]
                and passes["input_offset_pass"]
                and passes["icmr_pass"]
                and vgscheck == "No values found where vgs - vth < 0."
            ):
                converged = True

            sizing_Question = f"Currently,{opti_output}" + self.context.sizing_question
            pass_or_not = (
                f"gain_pass:{passes['gain_pass']},tr_gain_pass:{passes['tr_gain_pass']},"
                f"output_swing_pass:{passes['ow_pass']},input_offset_pass:{passes['input_offset_pass']}, "
                f"icmr_pass:{passes['icmr_pass']}, unity_bandwidth_pass:{passes['ubw_pass']}, "
                f"phase_margin_pass:{passes['pm_pass']}, power_pass:{passes['pr_pass']}, "
                f"cmrr_pass:{passes['cmrr_pass']} , thd_pass:{passes['thd_pass']}"
            )
            iteration += 1
            previous_results_list.append(f"Currently, {opti_output}, {pass_or_not},the netlist is {opti_netlist}")
            if len(previous_results_list) > 5:
                previous_results_list.pop(0)

            _write_history(self.config.history_file, previous_results_list)

            print(
                f"gain_target:{targets['gain_target']}, tr_gain_target:{targets['tr_gain_target']},"
                f"output_swing_target:{targets['output_swing_target']}, input_offset_target:{targets['input_offset_target']}, "
                f"icmr_target:{targets['icmr_target']}, unity_bandwidth_target:{targets['unity_bandwidth_target']}, "
                f"phase_margin_target:{targets['phase_margin_target']}, power_target:{targets['pr_target']}, "
                f"cmrr_target:{targets['cmrr_target']}, thd_target:{targets['thd_target']}"
            )
            print(
                f"gain_pass:{passes['gain_pass']},tr_gain_pass:{passes['tr_gain_pass']},"
                f"output_swing_pass:{passes['ow_pass']},input_offset_pass:{passes['input_offset_pass']}, "
                f"icmr_pass:{passes['icmr_pass']}, unity_bandwidth_pass:{passes['ubw_pass']}, "
                f"phase_margin_pass:{passes['pm_pass']}, power_pass:{passes['pr_pass']}, "
                f"cmrr_pass:{passes['cmrr_pass']} , thd_pass:{passes['thd_pass']}"
            )

        self._write_csv_results(initial_metrics, lists)

        return (
            {
                "converged": converged,
                "iterations": iteration,
                "gain_output": metrics["gain_output"] if "metrics" in locals() else None,
                "tr_gain_output": metrics["tr_gain_output"] if "metrics" in locals() else None,
                "output_swing_output": metrics["ow_output"] if "metrics" in locals() else None,
                "input_offset_output": metrics["offset_output"] if "metrics" in locals() else None,
                "ubw_output": metrics["ubw_output"] if "metrics" in locals() else None,
                "pm_output": metrics["pm_output"] if "metrics" in locals() else None,
                "pr_output": metrics["pr_output"] if "metrics" in locals() else None,
                "cmrr_output": metrics["cmrr_output"] if "metrics" in locals() else None,
                "icmr_output": metrics["icmr_output"] if "metrics" in locals() else None,
                "thd_output": metrics["thd_output"] if "metrics" in locals() else None,
            },
            opti_netlist,
        )
