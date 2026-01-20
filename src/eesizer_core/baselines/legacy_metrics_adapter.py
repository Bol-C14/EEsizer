from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import importlib
import os
import re
import sys
import time

from ..contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, RunResult, StrategyConfig
from ..contracts.enums import SimKind, StopReason
from ..contracts.errors import MetricError, SimulationError
from ..contracts.guards import GuardCheck, GuardReport
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..contracts.strategy import Strategy
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ..operators.guards import BehaviorGuardOperator, GuardChainOperator
from ..operators.netlist import TopologySignatureOperator
from ..runtime.recorder import RunRecorder
from ..runtime.recording_utils import (
    attempt_record,
    finalize_run,
    guard_failures,
    guard_report_to_dict,
    metrics_to_dict,
    param_space_to_dict,
    record_history_entry,
    record_operator_result,
    spec_to_dict,
    strategy_cfg_to_dict,
)
from ..sim.ngspice_runner import resolve_ngspice_executable, _probe_ngspice_version
from ..strategies.objective_eval import evaluate_objectives
from ..metrics.aliases import canonicalize_metrics


@dataclass(frozen=True)
class LegacyModules:
    simulation_utils: Any
    metrics_contract: Any | None = None


@dataclass(frozen=True)
class LegacyMetricHandler:
    sim: SimKind
    unit: str
    compute: Any


def find_legacy_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Find <repo>/legacy by walking upwards until legacy/legacy_eesizer exists."""
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        cand = p / "legacy"
        if (cand / "legacy_eesizer").is_dir():
            return cand
    return None


def ensure_legacy_importable(start: Optional[Path] = None) -> bool:
    """Ensure legacy_eesizer is importable by injecting <repo>/legacy into sys.path."""
    legacy_dir = find_legacy_dir(start=start)
    if legacy_dir is None:
        return False

    legacy_dir_str = str(legacy_dir)
    if legacy_dir_str not in sys.path:
        sys.path.insert(0, legacy_dir_str)
        importlib.invalidate_caches()

    try:
        legacy_pkg = importlib.import_module("legacy_eesizer")
        sys.modules.setdefault("agent_test_gpt", legacy_pkg)
        return True
    except Exception:
        return False


def _load_legacy_modules() -> LegacyModules:
    try:
        sim_utils = importlib.import_module("agent_test_gpt.simulation_utils")
        metrics_contract = importlib.import_module("agent_test_gpt.metrics_contract")
        return LegacyModules(simulation_utils=sim_utils, metrics_contract=metrics_contract)
    except Exception:
        if not ensure_legacy_importable():
            raise RuntimeError(
                "Legacy EEsizer modules not available: cannot import legacy_eesizer. "
                "Expected <repo>/legacy/legacy_eesizer to exist."
            )
        sim_utils = importlib.import_module("agent_test_gpt.simulation_utils")
        metrics_contract = importlib.import_module("agent_test_gpt.metrics_contract")
        return LegacyModules(simulation_utils=sim_utils, metrics_contract=metrics_contract)


def _resolve_metric_map(spec: CircuitSpec, cfg: StrategyConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    spec_map = spec.notes.get("legacy_metric_map")
    if isinstance(spec_map, Mapping):
        mapping.update({str(k): str(v) for k, v in spec_map.items()})
    cfg_map = cfg.notes.get("legacy_metric_map")
    if isinstance(cfg_map, Mapping):
        mapping.update({str(k): str(v) for k, v in cfg_map.items()})
    return mapping


def _build_handlers(sim_utils: Any, metrics_contract: Any | None) -> dict[str, LegacyMetricHandler]:
    handlers: dict[str, LegacyMetricHandler] = {}

    def _unit_from_contract(name: str, default: str) -> str:
        if metrics_contract is None:
            return default
        spec = getattr(metrics_contract, "METRICS", {}).get(name)
        return getattr(spec, "unit", default)

    def _register(metric: str, sim: SimKind, default_unit: str, fn_name: str) -> None:
        fn = getattr(sim_utils, fn_name, None)
        if fn is None:
            return
        handlers[metric] = LegacyMetricHandler(
            sim=sim,
            unit=_unit_from_contract(metric, default_unit),
            compute=fn,
        )

    _register("gain_db", SimKind.ac, "dB", "ac_gain")
    _register("bw_3db_hz", SimKind.ac, "Hz", "bandwidth")
    _register("ugbw_hz", SimKind.ac, "Hz", "unity_bandwidth")
    _register("phase_margin", SimKind.ac, "deg", "phase_margin")
    _register("pm_deg", SimKind.ac, "deg", "phase_margin")
    _register("out_swing_v", SimKind.dc, "V", "out_swing")
    _register("offset_v", SimKind.dc, "V", "offset")
    _register("icmr_v", SimKind.dc, "V", "ICMR")
    _register("tran_gain_db", SimKind.tran, "dB", "tran_gain")
    _register("power_w", SimKind.tran, "W", "stat_power")

    return handlers


def _coerce_str_list(value: Any, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return list(default)


def _run_legacy_simulation(
    sim_utils: Any,
    kind: SimKind,
    netlist_text: str,
    output_dir: Path,
    source_names: list[str],
    output_nodes: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if kind == SimKind.ac:
        netlist = sim_utils.ac_simulation(netlist_text, source_names, output_nodes, str(output_dir))
    elif kind == SimKind.dc:
        netlist = sim_utils.dc_simulation(netlist_text, source_names, output_nodes, str(output_dir))
    elif kind == SimKind.tran:
        netlist = sim_utils.trans_simulation(netlist_text, source_names, output_nodes, str(output_dir))
    else:
        raise SimulationError(f"Unsupported legacy sim kind: {kind}")

    netlist = _inject_dummy_print(netlist, kind, output_nodes[0] if output_nodes else "out")

    ok = sim_utils.run_ngspice(netlist, "netlist", output_dir=str(output_dir))
    if not ok:
        raise SimulationError(f"legacy ngspice failed for {kind.value}")


def _inject_dummy_print(netlist: str, kind: SimKind, node: str) -> str:
    """Ensure a .print directive exists so ngspice batch runs return success."""
    if re.search(r"(?im)^\s*\.print\b", netlist):
        return netlist

    node_name = node or "out"
    directive = f".print {kind.value} v({node_name})\n"

    m = re.search(r"(?im)^\s*\.control\b", netlist)
    if m:
        return netlist[: m.start()] + directive + netlist[m.start() :]

    m = re.search(r"(?im)^\s*\.end\b", netlist)
    if m:
        return netlist[: m.start()] + directive + netlist[m.start() :]

    return directive + netlist


def _record_legacy_provenance(
    recorder: RunRecorder | None,
    start_time: float,
    end_time: float,
    netlist_text: str,
    metric_names: list[str],
    metrics_payload: dict[str, float | None],
    notes: dict[str, Any],
) -> None:
    if recorder is None:
        return
    payload = {
        "operator": "legacy_metrics_baseline",
        "version": "0.1.0",
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": max(0.0, end_time - start_time),
        "command": "legacy_metrics_baseline",
        "inputs": {
            "netlist_text": stable_hash_str(netlist_text),
            "metric_names": stable_hash_json(metric_names),
        },
        "outputs": {"metrics": stable_hash_json(metrics_payload)},
        "notes": dict(notes),
    }
    recorder.append_jsonl("provenance/operator_calls.jsonl", payload)


@dataclass
class LegacyMetricsBaseline(Strategy):
    """Legacy metrics baseline run using legacy simulation_utils."""

    name: str = "baseline_legacy_metrics"
    version: str = "0.1.0"

    signature_op: Any = None
    behavior_guard_op: Any = None
    guard_chain_op: Any = None
    legacy_modules: LegacyModules | None = None

    def __post_init__(self) -> None:
        if self.signature_op is None:
            self.signature_op = TopologySignatureOperator()
        if self.behavior_guard_op is None:
            self.behavior_guard_op = BehaviorGuardOperator()
        if self.guard_chain_op is None:
            self.guard_chain_op = GuardChainOperator()

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
        history: list[dict[str, Any]] = []
        recorder: RunRecorder | None = None
        manifest = None
        if hasattr(ctx, "recorder") and hasattr(ctx, "manifest"):
            try:
                recorder = ctx.recorder()
                manifest = ctx.manifest()
            except Exception:
                recorder = None
                manifest = None

        sig_result = self.signature_op.run(
            {
                "netlist_text": source.text,
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
            ctx=None,
        )
        record_operator_result(recorder, sig_result)
        sig_res = sig_result.outputs
        circuit_ir = sig_res["circuit_ir"]
        signature = sig_res["signature"]

        rules = ParamInferenceRules(**cfg.notes.get("param_rules", {}))
        param_space = infer_param_space_from_ir(
            circuit_ir,
            rules=rules,
            frozen_param_ids=cfg.notes.get("frozen_param_ids", ()),
        )
        guard_cfg = dict(cfg.notes.get("guard_cfg", {}))
        guard_cfg.setdefault("wl_ratio_min", rules.wl_ratio_min)
        guard_cfg.setdefault("max_mul_factor", cfg.notes.get("max_mul_factor", 10.0))
        guard_cfg.setdefault("max_patch_ops", cfg.notes.get("max_patch_ops", 20))
        if "max_add_delta" in cfg.notes and "max_add_delta" not in guard_cfg:
            guard_cfg["max_add_delta"] = cfg.notes.get("max_add_delta")

        if manifest is not None:
            manifest.environment.setdefault("strategy_name", self.name)
            manifest.environment.setdefault("strategy_version", self.version)
            spec_payload = spec_to_dict(spec)
            param_payload = param_space_to_dict(param_space)
            cfg_payload = strategy_cfg_to_dict(cfg, guard_cfg)
            manifest.inputs.update(
                {
                    "netlist_sha256": stable_hash_str(source.text),
                    "spec_sha256": stable_hash_json(spec_payload),
                    "param_space_sha256": stable_hash_json(param_payload),
                    "cfg_sha256": stable_hash_json(cfg_payload),
                    "signature": signature,
                }
            )
            manifest.files.setdefault("inputs/source.sp", "inputs/source.sp")
            manifest.files.setdefault("inputs/spec.json", "inputs/spec.json")
            manifest.files.setdefault("inputs/param_space.json", "inputs/param_space.json")
            manifest.files.setdefault("inputs/cfg.json", "inputs/cfg.json")
            manifest.files.setdefault("inputs/signature.txt", "inputs/signature.txt")
            manifest.files.setdefault("history/iterations.jsonl", "history/iterations.jsonl")
            manifest.files.setdefault("history/summary.json", "history/summary.json")
            manifest.files.setdefault("provenance/operator_calls.jsonl", "provenance/operator_calls.jsonl")
            manifest.files.setdefault("best/best.sp", "best/best.sp")
            manifest.files.setdefault("best/best_metrics.json", "best/best_metrics.json")
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        sim_utils = None
        handlers: dict[str, LegacyMetricHandler] = {}
        metric_map = _resolve_metric_map(spec, cfg)
        metric_names = [obj.metric for obj in spec.objectives]
        legacy_names: list[str] = []
        name_lookup: dict[str, str] = {}
        for name in metric_names:
            legacy_name = metric_map.get(name, name)
            legacy_names.append(legacy_name)
            name_lookup[name] = legacy_name

        output_nodes = _coerce_str_list(
            spec.notes.get("output_nodes", cfg.notes.get("output_nodes")),
            ("out",),
        )
        source_names = _coerce_str_list(
            spec.notes.get("source_names", cfg.notes.get("source_names")),
            ("in",),
        )

        ngspice_path = resolve_ngspice_executable(cfg.notes.get("ngspice_bin"))
        prev_path = os.environ.get("NGSPICE_PATH")
        if ngspice_path:
            os.environ["NGSPICE_PATH"] = ngspice_path

        start_time = time.time()
        stage_map: dict[str, str] = {}
        warnings: list[str] = []
        errors: list[str] = []
        guard_report: GuardReport | None = None
        sim_runs = 0
        sim_runs_ok = 0
        sim_runs_failed = 0
        metrics_payload: dict[str, float | None] = {}
        metrics_bundle = MetricsBundle()
        try:
            legacy = self.legacy_modules or _load_legacy_modules()
            sim_utils = legacy.simulation_utils
            handlers = _build_handlers(sim_utils, legacy.metrics_contract)
            required_sims = {handlers[name].sim for name in legacy_names if name in handlers}
            for kind in sorted(required_sims, key=lambda k: k.value):
                stage_name = f"{kind.value}_i000_a00"
                stage_dir = Path(ctx.run_dir()) / stage_name
                _run_legacy_simulation(sim_utils, kind, source.text, stage_dir, source_names, output_nodes)
                stage_map[kind.value] = str(stage_dir)
                sim_runs += 1
                sim_runs_ok += 1

            for metric_name in metric_names:
                legacy_name = name_lookup[metric_name]
                handler = handlers.get(legacy_name)
                if handler is None:
                    metrics_payload[metric_name] = None
                    metrics_bundle.values[metric_name] = MetricValue(
                        name=metric_name,
                        value=None,
                        unit="",
                    )
                    continue
                stage_dir = stage_map.get(handler.sim.value)
                if stage_dir is None:
                    metrics_payload[metric_name] = None
                    metrics_bundle.values[metric_name] = MetricValue(
                        name=metric_name,
                        value=None,
                        unit=handler.unit,
                    )
                    continue
                if handler.sim == SimKind.ac:
                    base_name = "output_ac"
                elif handler.sim == SimKind.dc:
                    base_name = "output_dc"
                else:
                    base_name = "output_tran"
                value = handler.compute(base_name, output_dir=stage_dir)
                metrics_payload[metric_name] = value
                metrics_bundle.values[metric_name] = MetricValue(
                    name=metric_name,
                    value=value,
                    unit=handler.unit,
                )
        except (SimulationError, MetricError, RuntimeError, ValueError, ImportError, ModuleNotFoundError) as exc:
            sim_runs += 1
            sim_runs_failed += 1
            check = GuardCheck(
                name="behavior_guard",
                ok=False,
                severity="hard",
                reasons=(str(exc),),
                data={"error_type": type(exc).__name__},
            )
            guard_res = self.guard_chain_op.run({"checks": [check]}, ctx=None)
            record_operator_result(recorder, guard_res)
            guard_report = guard_res.outputs["report"]
            errors = guard_failures(guard_report)
        finally:
            if prev_path is None:
                os.environ.pop("NGSPICE_PATH", None)
            else:
                os.environ["NGSPICE_PATH"] = prev_path

        end_time = time.time()

        if guard_report is None:
            behavior_res = self.behavior_guard_op.run(
                {"metrics": metrics_bundle, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
                ctx=None,
            )
            record_operator_result(recorder, behavior_res)
            behavior_check = behavior_res.outputs["check"]
            guard_res = self.guard_chain_op.run({"checks": [behavior_check]}, ctx=None)
            record_operator_result(recorder, guard_res)
            guard_report = guard_res.outputs["report"]
            errors = guard_failures(guard_report)

        eval0 = evaluate_objectives(spec, metrics_bundle)
        history.append(
            {
                "iteration": 0,
                "patch": None,
                "signature_before": signature,
                "signature_after": signature,
                "metrics": {k: v.value for k, v in metrics_bundle.values.items()},
                "score": eval0["score"],
                "all_pass": eval0["all_pass"],
                "improved": False,
                "objectives": eval0["per_objective"],
                "sim_stages": stage_map,
                "warnings": warnings,
                "errors": errors,
                "guard": guard_report_to_dict(guard_report) if guard_report else None,
                "attempts": [attempt_record(0, None, guard_report, stage_map, warnings)],
            }
        )
        record_history_entry(recorder, history[-1])

        provenance_notes = {"legacy_import_path": getattr(sim_utils, "__file__", "unavailable")}
        if ngspice_path and sim_utils is not None:
            provenance_notes["ngspice_path"] = ngspice_path
            try:
                version = _probe_ngspice_version(ngspice_path)
                if version:
                    provenance_notes["ngspice_version"] = version
            except Exception as exc:
                provenance_notes["ngspice_version_error"] = str(exc)
        _record_legacy_provenance(
            recorder,
            start_time=start_time,
            end_time=end_time,
            netlist_text=source.text,
            metric_names=metric_names,
            metrics_payload=metrics_payload,
            notes=provenance_notes,
        )

        recording_errors = finalize_run(
            recorder=recorder,
            manifest=manifest,
            best_source=source,
            best_metrics=metrics_bundle,
            history=history,
            stop_reason=StopReason.baseline_legacy,
            best_score=eval0["score"],
            best_iter=0,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            best_metrics_payload=canonicalize_metrics(metrics_to_dict(metrics_bundle)),
        )

        return RunResult(
            best_source=source,
            best_metrics=metrics_bundle,
            history=history,
            stop_reason=StopReason.baseline_legacy,
            notes={
                "best_score": eval0["score"],
                "all_pass": eval0["all_pass"],
                "recording_errors": recording_errors,
            },
        )
