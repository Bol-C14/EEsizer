from __future__ import annotations

from pathlib import Path

from eesizer_core.contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    MetricValue,
    Objective,
    Patch,
    PatchOp,
)
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.strategies import PatchLoopStrategy


def _mock_measure_fn(_, iter_idx: int) -> MetricsBundle:
    """Mock metrics that improve on iteration 1, then plateau."""
    mb = MetricsBundle()
    value = -40.0 if iter_idx == 0 else -10.0
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def main() -> None:
    netlist = "M1 d g s b nmos W=1u L=0.1u\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    patches = [Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=1.1, why="mock"),))]
    policy = FixedSequencePolicy(patches=patches)

    strategy = PatchLoopStrategy(policy=policy, measure_fn=_mock_measure_fn)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=2, no_improve_patience=1))
    ctx = RunContext(workspace_root=Path(__file__).parent / "output")

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    print(f"stop_reason: {result.stop_reason}")
    print(f"best_score: {result.notes.get('best_score')}")
    print(f"history_len: {len(result.history)}")


if __name__ == "__main__":
    main()
