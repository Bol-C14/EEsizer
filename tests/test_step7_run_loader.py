from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.run_loader import RunLoader
from eesizer_core.strategies import PatchLoopStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_run_loader_reads_manifest_history_and_best(tmp_path):
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    policy = FixedSequencePolicy(patches=[patch])

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        return _make_metrics(-40.0) if iter_idx == 0 else _make_metrics(-10.0)

    strategy = PatchLoopStrategy(policy=policy, measure_fn=measure_fn)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=2, no_improve_patience=1))
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    loader = RunLoader(run_dir)
    manifest = loader.load_manifest()
    assert manifest.get("run_id")
    history = list(loader.iter_history())
    assert history
    best = loader.load_best()
    assert "M1" in best["best_sp"]
