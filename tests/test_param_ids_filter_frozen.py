import json
from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import CornerSearchStrategy, GridSearchStrategy


def _make_metrics() -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=-10.0, unit="dB")
    return mb


def _load_candidates(path: Path) -> list[dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_grid_search_filters_frozen_by_default(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        return _make_metrics()

    strategy = GridSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "frozen_param_ids": ["r1.value"],
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)
    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    candidates = _load_candidates(ctx.run_dir() / "search" / "candidates.json")
    assert candidates
    for cand in candidates:
        assert "r1.value" not in cand


def test_grid_search_respects_explicit_param_ids(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        return _make_metrics()

    strategy = GridSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "frozen_param_ids": ["r1.value"],
            "grid_search": {
                "mode": "coordinate",
                "levels": 2,
                "span_mul": 2.0,
                "scale": "linear",
                "param_ids": ["r1.value"],
                "allow_param_ids_override_frozen": True,
            },
        },
    )
    ctx = RunContext(workspace_root=tmp_path)
    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    candidates = _load_candidates(ctx.run_dir() / "search" / "candidates.json")
    assert candidates
    assert all("r1.value" in cand for cand in candidates)


def test_corner_search_filters_frozen_by_default(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        return _make_metrics()

    strategy = CornerSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "frozen_param_ids": ["r1.value"],
            "corner_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)
    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    corner_set = json.loads((ctx.run_dir() / "search" / "corner_set.json").read_text(encoding="utf-8"))
    assert "r1.value" not in corner_set.get("search_param_ids", [])


def test_corner_search_respects_explicit_param_ids(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        return _make_metrics()

    strategy = CornerSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "frozen_param_ids": ["r1.value"],
            "corner_search": {
                "mode": "coordinate",
                "levels": 2,
                "span_mul": 2.0,
                "scale": "linear",
                "param_ids": ["r1.value"],
                "allow_param_ids_override_frozen": True,
            },
        },
    )
    ctx = RunContext(workspace_root=tmp_path)
    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    corner_set = json.loads((ctx.run_dir() / "search" / "corner_set.json").read_text(encoding="utf-8"))
    assert "r1.value" in corner_set.get("search_param_ids", [])
