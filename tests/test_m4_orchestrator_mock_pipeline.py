from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import MultiAgentOrchestratorStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_orchestrator_runs_grid_search_subrun_and_writes_outputs(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec()  # orchestrator will synthesize

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        return _make_metrics(-40.0) if iter_idx == 0 else _make_metrics(-10.0)

    strat = MultiAgentOrchestratorStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=5, no_improve_patience=1), notes={})
    ctx = RunContext(workspace_root=tmp_path)

    result = strat.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    # Orchestrator run outputs
    run_dir = ctx.run_dir()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "orchestrator" / "plan.json").exists()
    assert (run_dir / "orchestrator" / "artifacts" / "index.json").exists()

    # Child search run exists
    child = result.notes.get("child_run", {})
    child_run_dir = child.get("run_dir")
    assert child_run_dir
    child_path = tmp_path / "runs" / child["run_id"]
    assert child_path.exists()
    assert (child_path / "best" / "best_metrics.json").exists()
    assert result.best_metrics.get("ac_mag_db_at_1k").value == -10.0


def test_orchestrator_can_switch_to_corner_search(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec()

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        return _make_metrics(-30.0) if iter_idx == 0 else _make_metrics(-15.0)

    strat = MultiAgentOrchestratorStrategy(measure_fn=measure_fn)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=5, no_improve_patience=1),
        notes={"orchestrator": {"robust": True}, "corner_search": {"levels": 2, "span_mul": 2.0, "scale": "linear"}},
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strat.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    child = result.notes.get("child_run", {})
    assert child.get("strategy") == "corner_search"