from pathlib import Path

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective, SourceKind
from eesizer_core.contracts.enums import StopReason
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import PatchLoopStrategy
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable


@pytest.mark.integration
def test_patch_loop_strategy_integration_with_ngspice(tmp_path):
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    netlist_path = examples_dir / "rc_lowpass.sp"
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    patch = Patch(ops=(PatchOp(param="r1.value", op=PatchOpType.mul, value=1.1, why="nudge"),))
    policy = FixedSequencePolicy(patches=[patch])

    ctx = RunContext(workspace_root=tmp_path)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=2, no_improve_patience=1))

    strategy = PatchLoopStrategy(policy=policy)
    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    assert result.history
    assert result.stop_reason in {StopReason.policy_stop, StopReason.reached_target, StopReason.no_improvement}
    assert result.best_metrics.values.get("ac_mag_db_at_1k") is not None
    # ensure run outputs folder exists
    assert (ctx.run_dir()).exists()
