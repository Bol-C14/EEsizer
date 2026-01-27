from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import GridSearchStrategy


def _make_metrics() -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=0.0, unit="dB")
    return mb


def test_grid_search_can_continue_after_baseline_pass(tmp_path) -> None:
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    def measure_fn(_, __: int) -> MetricsBundle:
        return _make_metrics()

    strategy = GridSearchStrategy(measure_fn=measure_fn)

    cfg_stop = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )
    ctx_stop = RunContext(workspace_root=tmp_path / "stop")
    result_stop = strategy.run(spec=spec, source=source, ctx=ctx_stop, cfg=cfg_stop)
    assert len(result_stop.history) == 1  # baseline only

    cfg_continue = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
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
    ctx_continue = RunContext(workspace_root=tmp_path / "cont")
    result_continue = strategy.run(spec=spec, source=source, ctx=ctx_continue, cfg=cfg_continue)
    assert len(result_continue.history) > 1
    assert any(entry.get("iteration") == 1 and entry.get("candidate") for entry in result_continue.history)
