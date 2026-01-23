from eesizer_core.contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    MetricValue,
    Objective,
    PatchOp,
    Patch,
)
from eesizer_core.contracts.enums import PatchOpType, SourceKind, StopReason
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.strategies import PatchLoopStrategy


def _make_metrics(value: float) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=value, unit="dB")
    return mb


def test_patch_loop_strategy_with_measure_fn_updates_best_and_stops_on_policy():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    # Policy: one patch then stop
    patch1 = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=1.1),))
    policy = FixedSequencePolicy(patches=[patch1])

    # measure_fn returns improving metrics on first iteration
    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        if iter_idx == 0:
            return _make_metrics(-40.0)
        return _make_metrics(-10.0)

    strategy = PatchLoopStrategy(policy=policy, measure_fn=measure_fn)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3, no_improve_patience=2))

    class DummyCtx:
        def run_dir(self):
            return None

    result = strategy.run(spec=spec, source=source, ctx=DummyCtx(), cfg=cfg)

    assert result.stop_reason == StopReason.policy_stop or result.stop_reason == StopReason.reached_target
    assert result.best_metrics.values["ac_mag_db_at_1k"].value == -10.0
    assert len(result.history) >= 2
    # patched source should reflect the applied patch
    assert "1.1" in result.best_source.text or "1.1e-06" in result.best_source.text
    assert "objectives" in result.history[-1]


def test_patch_loop_strategy_policy_stop_does_not_crash_when_target_not_reached():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=0.0, sense="ge"),))

    policy = FixedSequencePolicy(patches=[])

    def measure_fn(_, iter_idx: int) -> MetricsBundle:
        mb = MetricsBundle()
        mb.values["ac_mag_db_at_1k"] = MetricValue(name="ac_mag_db_at_1k", value=-40.0, unit="dB")
        return mb

    strategy = PatchLoopStrategy(policy=policy, measure_fn=measure_fn)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3, no_improve_patience=2))

    class DummyCtx:
        def run_dir(self):
            return None

    result = strategy.run(spec=spec, source=source, ctx=DummyCtx(), cfg=cfg)
    assert result.stop_reason == StopReason.policy_stop
    assert len(result.history) == 2  # iter0 baseline + iter1 stop record
    assert result.history[-1]["patch"]["stop"] is True
    assert result.history[-1]["signature_after"] == result.history[-1]["signature_before"]
