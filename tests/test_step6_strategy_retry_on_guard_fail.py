from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType, SourceKind, StopReason
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.strategies import PatchLoopStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_strategy_retries_on_guard_fail_and_recovers():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    bad_patch = Patch(ops=(PatchOp(param="m2.w", op=PatchOpType.set, value="2u"),))
    ok_patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    policy = FixedSequencePolicy(patches=[bad_patch, ok_patch])

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        return _make_metrics(-40.0) if iter_idx == 0 else _make_metrics(-10.0)

    strategy = PatchLoopStrategy(policy=policy, measure_fn=measure_fn, max_patch_retries=1)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=2, no_improve_patience=1))

    class DummyCtx:
        def run_dir(self):
            return None

    result = strategy.run(spec=spec, source=source, ctx=DummyCtx(), cfg=cfg)

    assert result.stop_reason in {StopReason.reached_target, StopReason.policy_stop}
    assert len(result.history) >= 2
    iter1 = result.history[1]
    assert "guard" in iter1
    assert "attempts" in iter1
    assert any(attempt["guard"] and attempt["guard"]["ok"] is False for attempt in iter1["attempts"])
