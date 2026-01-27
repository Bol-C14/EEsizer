import json

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators import CornerValidateOperator
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import GridSearchStrategy


def test_step5_corner_validate_writes_robust_files_and_updates_report(tmp_path):
    pytest.importorskip("matplotlib")

    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(
        objectives=(
            Objective(metric="ugbw_hz", target=1e6, sense="ge"),
            Objective(metric="phase_margin_deg", target=60.0, sense="ge"),
            Objective(metric="power_w", target=1e-3, sense="le"),
        )
    )

    def measure_fn(src: CircuitSource, iter_idx: int) -> MetricsBundle:
        # Deterministic but depends on iteration so candidates differ.
        mb = MetricsBundle()
        mb.values["ugbw_hz"] = MetricValue(name="ugbw_hz", value=1e6 + 1e5 * iter_idx, unit="Hz")
        mb.values["phase_margin_deg"] = MetricValue(name="phase_margin_deg", value=70.0, unit="deg")
        mb.values["power_w"] = MetricValue(name="power_w", value=5e-4 + 1e-5 * iter_idx, unit="W")
        return mb

    ctx = RunContext(workspace_root=tmp_path)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=4, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "grid_search": {
                "mode": "coordinate",
                "levels": 2,
                "span_mul": 2.0,
                "scale": "linear",
                "continue_after_baseline_pass": True,
            },
        },
    )

    grid = GridSearchStrategy(measure_fn=measure_fn)
    grid.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    op = CornerValidateOperator(measure_fn=measure_fn)
    op.run(
        {
            "run_dir": ctx.run_dir(),
            "recorder": ctx.recorder(),
            "manifest": ctx.manifest(),
            "source": source,
            "spec": spec,
            "cfg": cfg,
            "candidates_source": "topk",
            "corner_validate": {"corners": "global", "override_mode": "set"},
        },
        ctx=ctx,
    )

    run_dir = ctx.run_dir()
    assert (run_dir / "search" / "robust_candidates.json").exists()
    assert (run_dir / "search" / "robust_topk.json").exists()
    assert (run_dir / "search" / "robust_pareto.json").exists()
    assert (run_dir / "search" / "robust_meta.json").exists()

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Robustness Validation" in report

    # Robust plot should no longer be skipped and should be embedded.
    assert (run_dir / "plots" / "robust_nominal_vs_worst.png").exists()
    assert "![](plots/robust_nominal_vs_worst.png)" in report

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    assert "search/robust_topk.json" in files
    assert "search/robust_pareto.json" in files
