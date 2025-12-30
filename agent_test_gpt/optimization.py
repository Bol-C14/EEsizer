"""Tool calling and optimization runners."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from pathlib import Path
import time

import csv
import json
import os
import time

import numpy as np

from agent_test_gpt import prompts
from agent_test_gpt import config
from agent_test_gpt.simulation_utils import _ensure_flat_str_list
from agent_test_gpt.models import TargetValues
from agent_test_gpt.toolchain import validate_tool_chain


@dataclass
class ToolCallingContext:
    netlist: str
    source_names: List[str]
    output_nodes: List[str]
    output_dir: str


@dataclass
class ToolCallingConfig:
    input_txt: str
    filtered_txt: str
    output_csv: str
    output_txt: str
    output_dir: str

    @classmethod
    def from_output_dir(cls, output_dir: str = config.RUN_OUTPUT_ROOT) -> "ToolCallingConfig":
        base = Path(output_dir)
        return cls(
            input_txt=str(base / "op.txt"),
            filtered_txt=str(base / "vgscheck.txt"),
            output_csv=str(base / "vgscheck.csv"),
            output_txt=str(base / "vgscheck_output.txt"),
            output_dir=output_dir,
        )


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
        self.config = config or ToolCallingConfig.from_output_dir(context.output_dir)

    def _run_ngspice_with_vgscheck(self, sim_netlist: str) -> str:
        ok = self.deps.run_ngspice(sim_netlist, "netlist", output_dir=self.context.output_dir)
        if ok:
            self.deps.filter_lines(self.config.input_txt, self.config.filtered_txt)
            if self.deps.convert_to_csv(self.config.filtered_txt, self.config.output_csv):
                self.deps.format_csv_to_key_value(self.config.output_csv, self.config.output_txt)
                return self.deps.read_txt_as_string(self.config.output_txt)
            return "Vgs/Vth check not available (no parsed devices)."
        raise RuntimeError("NGspice run failed; halting tool chain.")

    def run(self, tool_chain: Dict) -> ToolCallingResult:
        validate_tool_chain(tool_chain)
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
        output_dir = self.context.output_dir

        source_names = _ensure_flat_str_list("context.source_names", self.context.source_names)
        output_nodes = _ensure_flat_str_list("context.output_nodes", self.context.output_nodes)

        for tool_call in tool_chain["tool_calls"]:
            tool_name = tool_call["name"].lower()
            if tool_name == "dc_simulation":
                sim_netlist = self.deps.dc_simulation(sim_netlist, source_names, output_nodes, output_dir)
            elif tool_name == "ac_simulation":
                sim_netlist = self.deps.ac_simulation(sim_netlist, source_names, output_nodes, output_dir)
                print(f"ac_netlist:{sim_netlist}")
            elif tool_name == "transient_simulation":
                sim_netlist = self.deps.trans_simulation(sim_netlist, source_names, output_nodes, output_dir)
            elif tool_name == "run_ngspice":
                vgscheck = self._run_ngspice_with_vgscheck(sim_netlist)
            elif tool_name == "ac_gain":
                gain = self.deps.ac_gain("output_ac", output_dir=output_dir)
                print(f"ac_gain result: {gain}")
            elif tool_name == "output_swing":
                ow = self.deps.out_swing("output_dc", output_dir=output_dir)
                print(f"output swing result: {ow}")
            elif tool_name == "icmr":
                icmr = self.deps.ICMR("output_dc", output_dir=output_dir)
                print(f"input common mode voltage result: {icmr}")
            elif tool_name == "offset":
                Offset = self.deps.offset("output_dc", output_dir=output_dir)
                print(f"input offset result: {Offset}")
            elif tool_name == "tran_gain":
                tr_gain = self.deps.tran_gain("output_tran", output_dir=output_dir)
                print(f"tran_gain result: {tr_gain}")
            elif tool_name == "bandwidth":
                bw = self.deps.bandwidth("output_ac", output_dir=output_dir)
                print(f"bandwidth result: {bw}")
            elif tool_name == "unity_bandwidth":
                ubw = self.deps.unity_bandwidth("output_ac", output_dir=output_dir)
                print(f"unity bandwidth result: {ubw}")
            elif tool_name == "phase_margin":
                pm = self.deps.phase_margin("output_ac", output_dir=output_dir)
                print(f"phase margin: {pm}")
            elif tool_name == "cmrr_tran":
                cmrr, cmrr_max = self.deps.cmrr_tran(sim_netlist, output_dir=output_dir)
                print(f"cmrr: {cmrr}, cmrr_max: {cmrr_max}")
            elif tool_name == "power":
                pr = self.deps.stat_power("output_tran", output_dir=output_dir)
                print(f"power: {pr}")
            elif tool_name == "thd_input_range":
                thd, ir = self.deps.thd_input_range("output_tran", output_dir=output_dir)
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
    source_names: List[str]
    output_nodes: List[str]
    output_dir: str


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
    llm_delay_seconds: float = 0.0

    @classmethod
    def from_output_dir(cls, output_dir: str) -> "OptimizationConfig":
        base = Path(output_dir)
        return cls(
            max_iterations=config.MAX_ITERATIONS,
            tolerance=config.TOLERANCE,
            output_dir=output_dir,
            history_file=str(base / "result_history.txt"),
            csv_file=str(base / "metrics.csv"),
            input_txt=str(base / "op.txt"),
            filtered_txt=str(base / "vgscheck.txt"),
            output_csv=str(base / "vgscheck.csv"),
            output_txt=str(base / "vgscheck_output.txt"),
            llm_delay_seconds=0.0,
        )


@dataclass
class OptimizationDeps:
    make_chat_completion_request: Callable
    sanitize_netlist: Callable
    dc_simulation: Callable
    ac_simulation: Callable
    trans_simulation: Callable
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


def parse_target_values(target_values: str, extracting_method: Callable) -> TargetValues:
    """Parse target values JSON from LLM into structured targets and pass flags without globals."""
    cleaned = target_values.strip().strip("`").replace("json", "").strip()
    payload = json.loads(cleaned)
    if "target_values" not in payload or not isinstance(payload["target_values"], list):
        raise ValueError("target_values must contain a list under 'target_values'")

    merged: Dict[str, float | None] = {}
    for item in payload["target_values"]:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            merged[key] = value

    gain_target = extracting_method(merged.get("ac_gain_target", "0")) if "ac_gain_target" in merged else None
    bandwidth_target = extracting_method(merged.get("bandwidth_target", "0")) if "bandwidth_target" in merged else None
    unity_bandwidth_target = extracting_method(merged.get("unity_bandwidth_target", "0")) if "unity_bandwidth_target" in merged else None
    phase_margin_target = extracting_method(merged.get("phase_margin_target", "0")) if "phase_margin_target" in merged else None
    tr_gain_target = extracting_method(merged.get("transient_gain_target", "0")) if "transient_gain_target" in merged else None
    input_offset_target = extracting_method(merged.get("input_offset_target", "0")) if "input_offset_target" in merged else None
    output_swing_target = extracting_method(merged.get("output_swing_target", "0")) if "output_swing_target" in merged else None
    pr_target = extracting_method(merged.get("power_target", "0")) if "power_target" in merged else None
    cmrr_target = extracting_method(merged.get("cmrr_target", "0")) if "cmrr_target" in merged else None
    thd_raw = extracting_method(merged.get("thd_target", "0")) if "thd_target" in merged else None
    thd_target = -np.abs(thd_raw) if thd_raw is not None else None
    icmr_target = extracting_method(merged.get("input_common_mode_range_target", "0")) if "input_common_mode_range_target" in merged else None

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
        "gain_pass": gain_target is None,
        "tr_gain_pass": tr_gain_target is None,
        "dc_gain_pass": tr_gain_target is None,
        "ow_pass": output_swing_target is None,
        "bw_pass": bandwidth_target is None,
        "ubw_pass": unity_bandwidth_target is None,
        "pm_pass": phase_margin_target is None,
        "pr_pass": pr_target is None,
        "cmrr_pass": cmrr_target is None,
        "thd_pass": thd_target is None,
        "input_offset_pass": input_offset_target is None,
        "icmr_pass": icmr_target is None,
    }
    return TargetValues(targets=targets, passes=passes)


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
        if self.config.llm_delay_seconds:
            time.sleep(self.config.llm_delay_seconds)

        optimising_prompt = prompts.build_optimising_prompt(self.context.type_identified, analysis, previous_results)
        optimising = self.deps.make_chat_completion_request(optimising_prompt)
        print(optimising)
        if self.config.llm_delay_seconds:
            time.sleep(self.config.llm_delay_seconds)

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

        raw_netlist = self.deps.extract_code(modified_output)
        try:
            opti_netlist = self.deps.sanitize_netlist(raw_netlist)
        except Exception as exc:
            raise ValueError(f"LLM-produced netlist failed safety checks: {exc}") from exc
        print(opti_netlist)

        source_names = _ensure_flat_str_list("context.source_names", self.context.source_names)
        output_nodes = _ensure_flat_str_list("context.output_nodes", self.context.output_nodes)
        output_dir = self.context.output_dir

        for tool_call in tools["tool_calls"]:
            tool_name = tool_call["name"].lower()

            if tool_name == "dc_simulation":
                opti_netlist = self.deps.dc_simulation(opti_netlist, source_names, output_nodes, output_dir)
            elif tool_name == "ac_simulation":
                opti_netlist = self.deps.ac_simulation(opti_netlist, source_names, output_nodes, output_dir)
            elif tool_name == "transient_simulation":
                opti_netlist = self.deps.trans_simulation(opti_netlist, source_names, output_nodes, output_dir)
            elif tool_name == "run_ngspice":
                ok = self.deps.run_ngspice(opti_netlist, "netlist", output_dir=output_dir)
                print("running ngspice")
                if not ok:
                    raise RuntimeError("NGspice run failed during optimization")
                self.deps.filter_lines(self.config.input_txt, self.config.filtered_txt)
                self.deps.convert_to_csv(self.config.filtered_txt, self.config.output_csv)
                self.deps.format_csv_to_key_value(self.config.output_csv, self.config.output_txt)
                vgscheck = self.deps.read_txt_as_string(self.config.output_txt)
            elif tool_name == "ac_gain":
                gain_output = self.deps.ac_gain("output_ac", output_dir=output_dir)
            elif tool_name == "dc_gain":
                dc_gain_output = self.deps.dc_gain("output_dc", output_dir=output_dir)
            elif tool_name == "output_swing":
                ow_output = self.deps.out_swing("output_dc", output_dir=output_dir)
            elif tool_name == "offset":
                offset_output = self.deps.offset("output_dc", output_dir=output_dir)
            elif tool_name == "icmr":
                icmr_output = self.deps.ICMR("output_dc", output_dir=output_dir)
            elif tool_name == "tran_gain":
                tr_gain_output = self.deps.tran_gain("output_tran", output_dir=output_dir)
            elif tool_name == "bandwidth":
                bw_output = self.deps.bandwidth("output_ac", output_dir=output_dir)
            elif tool_name == "unity_bandwidth":
                ubw_output = self.deps.unity_bandwidth("output_ac", output_dir=output_dir)
            elif tool_name == "phase_margin":
                pm_output = self.deps.phase_margin("output_ac", output_dir=output_dir)
            elif tool_name == "power":
                pr_output = self.deps.stat_power("output_tran", output_dir=output_dir)
            elif tool_name == "thd_input_range":
                thd_output, _ir_output = self.deps.thd_input_range("output_tran", output_dir=output_dir)
            elif tool_name == "cmrr_tran":
                cmrr_output, cmrr_max = self.deps.cmrr_tran(opti_netlist, output_dir=output_dir)

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

        target_bundle = parse_target_values(target_values, extracting_method)
        targets = target_bundle.targets
        passes = target_bundle.passes
        opti_netlist = sim_netlist
        previous_results_list = self._initialize_history(opti_netlist)
        sizing_Question = f"Currently, {self.context.sim_output}. " + self.context.sizing_question

        while iteration < max_iterations and not converged:
            with open(self.config.history_file, "r") as f:
                previous_results = f.read()
            self.logger.info("Iteration %s", iteration)

            modified_output = self._run_llm_cycle(previous_results, sizing_Question, opti_netlist)
            self.logger.debug("Modified output: %s", modified_output)
            if self.config.llm_delay_seconds:
                time.sleep(self.config.llm_delay_seconds)

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
            self.logger.info("Opti output: %s", opti_output)

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

            self.logger.debug(
                "Targets: gain=%s tr_gain=%s ow=%s offset=%s icmr=%s ubw=%s pm=%s power=%s cmrr=%s thd=%s",
                targets['gain_target'],
                targets['tr_gain_target'],
                targets['output_swing_target'],
                targets['input_offset_target'],
                targets['icmr_target'],
                targets['unity_bandwidth_target'],
                targets['phase_margin_target'],
                targets['pr_target'],
                targets['cmrr_target'],
                targets['thd_target'],
            )
            self.logger.debug(
                "Pass flags: gain=%s tr_gain=%s ow=%s offset=%s icmr=%s ubw=%s pm=%s power=%s cmrr=%s thd=%s",
                passes['gain_pass'],
                passes['tr_gain_pass'],
                passes['ow_pass'],
                passes['input_offset_pass'],
                passes['icmr_pass'],
                passes['ubw_pass'],
                passes['pm_pass'],
                passes['pr_pass'],
                passes['cmrr_pass'],
                passes['thd_pass'],
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
