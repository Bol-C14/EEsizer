import json

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import GridSearchStrategy


def test_tradeoff_objectives_fallback_for_rc_like_specs(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(
        objectives=(
            Objective(metric="ac_mag_db_at_1k_vout", target=-20.0, sense="ge"),
            Objective(metric="dc_vout_last_vout", target=0.0, sense="ge"),
        )
    )

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        mb = MetricsBundle()
        mb.values["ac_mag_db_at_1k_vout"] = MetricValue(
            name="ac_mag_db_at_1k_vout", value=-40.0 + 2.0 * iter_idx, unit="dB"
        )
        mb.values["dc_vout_last_vout"] = MetricValue(name="dc_vout_last_vout", value=0.1 * iter_idx, unit="V")
        return mb

    strategy = GridSearchStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=4, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "plots" / "tradeoff_objectives.png").exists()
    data = json.loads((run_dir / "plots" / "tradeoff_objectives_data.json").read_text(encoding="utf-8"))
    assert data["x_metric"] == "dc_vout_last_vout"
    assert data["y_metric"] == "ac_mag_db_at_1k_vout"

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "![](plots/tradeoff_objectives.png)" in report

