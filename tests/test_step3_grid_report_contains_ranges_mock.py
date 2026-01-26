from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import GridSearchStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_grid_report_contains_ranges_and_candidates_meta(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        return _make_metrics(-40.0) if iter_idx == 0 else _make_metrics(-10.0)

    strategy = GridSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=4, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\\.value|c1\\.value)$"]},
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "search" / "ranges.json").exists()
    assert (run_dir / "search" / "candidates_meta.json").exists()

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Ranges & Discretization" in report
    assert "Candidate Generation" in report
    assert "Parameters Selected" in report
