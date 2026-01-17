from pathlib import Path

import pytest

from eesizer_core.analysis.compare_runs import compare_runs
from eesizer_core.baselines import LegacyMetricsBaseline
from eesizer_core.baselines.legacy_metrics_adapter import ensure_legacy_importable
from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import NoOptBaselineStrategy


@pytest.mark.integration
def test_compare_new_vs_legacy_metrics_integration(tmp_path):
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    pytest.importorskip("scipy")
    if not ensure_legacy_importable():
        pytest.skip("legacy_eesizer not importable; skip legacy comparison integration test")

    netlist = "V1 in 0 AC 2\nR1 in out 1k\nC1 out 0 1u\nRleak out 0 1G\n.end\n"
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist,
        metadata={"base_dir": Path(__file__).resolve().parent},
    )
    spec = CircuitSpec(
        objectives=(Objective(metric="ac_unity_gain_freq", target=None, sense="ge"),),
        notes={
            "output_nodes": ["out"],
            "source_names": ["in"],
            "legacy_metric_map": {"ac_unity_gain_freq": "ugbw_hz"},
            "compare_tol": {"ugbw_hz": {"rel": 0.5}},
        },
    )
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))

    ctx_a = RunContext(workspace_root=tmp_path / "run_a")
    ctx_b = RunContext(workspace_root=tmp_path / "run_b")

    NoOptBaselineStrategy().run(spec=spec, source=source, ctx=ctx_a, cfg=cfg)
    LegacyMetricsBaseline().run(spec=spec, source=source, ctx=ctx_b, cfg=cfg)

    out_dir = tmp_path / "comparison"
    comparison = compare_runs(ctx_a.run_dir(), ctx_b.run_dir(), out_dir)

    row = comparison["metrics"].get("ugbw_hz")
    assert row is not None
    assert row["a"] is not None
    assert row["b"] is not None
    assert row["within_tol"] is True
