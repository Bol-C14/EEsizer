from pathlib import Path
import json

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.domain.spice.patching import parse_scalar_numeric
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import CornerSearchStrategy


def _parse_value(text: str, device_name: str) -> float:
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0].lower() == device_name:
            return parse_scalar_numeric(parts[-1])
    raise ValueError(f"missing {device_name}")


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_corner_search_writes_outputs_and_corners(tmp_path: Path) -> None:
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-1e-9, sense="ge"),))

    def measure_fn(src: CircuitSource, _: int) -> MetricsBundle:
        r_val = _parse_value(src.text, "r1")
        c_val = _parse_value(src.text, "c1")
        return _make_metrics(-(r_val * c_val))

    strategy = CornerSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "corner_search": {"mode": "coordinate", "levels": 2, "span_mul": 10.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "history" / "iterations.jsonl").exists()
    assert (run_dir / "best" / "best.sp").exists()
    assert (run_dir / "best" / "best_metrics.json").exists()
    assert (run_dir / "search" / "corner_set.json").exists()
    assert (run_dir / "search" / "candidates.json").exists()
    assert (run_dir / "search" / "ranges.json").exists()
    assert (run_dir / "search" / "candidates_meta.json").exists()
    assert (run_dir / "search" / "topk.json").exists()
    assert (run_dir / "search" / "pareto.json").exists()
    assert (run_dir / "report.md").exists()

    ranges = json.loads((run_dir / "search" / "ranges.json").read_text(encoding="utf-8"))
    assert any(entry.get("param_id") == "r1.value" and entry.get("nominal") is not None for entry in ranges)

    assert result.history
    for entry in result.history:
        assert "corners" in entry

    assert result.history[0]["worst_corner_id"] == "c1.value_high"


def test_corner_search_does_not_gate_on_failed_baseline_corners(tmp_path: Path) -> None:
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-1e-9, sense="ge"),))

    def measure_fn(src: CircuitSource, _: int) -> MetricsBundle:
        r_val = _parse_value(src.text, "r1")
        c_val = _parse_value(src.text, "c1")
        return _make_metrics(-(r_val * c_val))

    strategy = CornerSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=2, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "guard_cfg": {"max_add_delta": 0.0},
            "corner_search": {"mode": "coordinate", "levels": 2, "span_mul": 10.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    assert len(result.history) > 1
