from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.strategies.interactive_session import InteractiveSessionStrategy


def test_step6_continue_skips_unchanged_grid_phase(tmp_path: Path) -> None:
    calls = {"n": 0}

    def measure_fn(_src: CircuitSource, iter_idx: int) -> MetricsBundle:
        calls["n"] += 1
        mb = MetricsBundle()
        # Deterministic: baseline passes, but we still want the grid stage to be exercised.
        mb.values["ugbw_hz"] = MetricValue(name="ugbw_hz", value=1e6 + 1e3 * iter_idx, unit="Hz")
        mb.values["phase_margin_deg"] = MetricValue(name="phase_margin_deg", value=70.0, unit="deg")
        mb.values["power_w"] = MetricValue(name="power_w", value=1e-3, unit="W")
        return mb

    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\nC1 out 0 1p\n.end\n")
    spec = CircuitSpec(
        objectives=(
            Objective(metric="ugbw_hz", target=1e6, sense="ge"),
            Objective(metric="phase_margin_deg", target=60.0, sense="ge"),
            Objective(metric="power_w", target=2e-3, sense="le"),
        )
    )
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        seed=0,
        notes={
            "session": {"run_baseline": False},
            "grid_search": {
                "mode": "coordinate",
                "levels": 2,
                "span_mul": 2.0,
                "scale": "linear",
                "continue_after_baseline_pass": True,
            },
        },
    )

    ws = tmp_path / "ws"
    strat = InteractiveSessionStrategy(measure_fn=measure_fn)
    session_ctx = strat.start_session(
        bench_id="rc",
        source=source,
        spec=spec,
        cfg=cfg,
        workspace_root=ws,
        run_to_phase="p1_grid",
        reason="unit_test",
    )

    calls_after_start = calls["n"]
    assert calls_after_start > 0

    strat.continue_session(
        session_run_dir=session_ctx.run_dir(),
        source=source,
        next_phase="p1_grid",
        reason="unit_test_continue_no_delta",
    )

    assert calls["n"] == calls_after_start

    # Checkpoints + meta report exist.
    assert (session_ctx.run_dir() / "session" / "checkpoints" / "p1_grid.json").exists()
    assert (session_ctx.run_dir() / "session" / "meta_report.md").exists()

