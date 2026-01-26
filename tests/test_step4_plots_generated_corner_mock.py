import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import CornerSearchStrategy


def test_plots_generated_for_corner_search(tmp_path):
    pytest.importorskip("matplotlib")

    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        mb = MetricsBundle()
        mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=-10.0, unit="dB")
        mb.values["ugbw_hz"] = MetricValue(name="ugbw_hz", value=2e6, unit="Hz")
        mb.values["phase_margin_deg"] = MetricValue(name="phase_margin_deg", value=70.0, unit="deg")
        mb.values["power_w"] = MetricValue(name="power_w", value=2e-3, unit="W")
        return mb

    strategy = CornerSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "corner_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "plots" / "index.json").exists()
    assert (run_dir / "plots" / "robust_nominal_vs_worst.png").exists()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "robust_nominal_vs_worst.png" in report
