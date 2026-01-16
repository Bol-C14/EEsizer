from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import PatchLoopStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_patch_loop_writes_run_outputs(tmp_path):
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

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "inputs" / "source.sp").exists()
    assert (run_dir / "inputs" / "spec.json").exists()
    assert (run_dir / "history" / "iterations.jsonl").exists()
    assert (run_dir / "provenance" / "operator_calls.jsonl").exists()
    assert (run_dir / "best" / "best.sp").exists()
    assert (run_dir / "best" / "best_metrics.json").exists()

    lines = (run_dir / "history" / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.history)
