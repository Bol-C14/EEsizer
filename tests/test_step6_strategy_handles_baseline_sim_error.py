from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SimKind, SourceKind, StopReason
from eesizer_core.contracts.errors import SimulationError
from eesizer_core.contracts.operators import OperatorResult
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.sim.artifacts import RawSimData, SpiceDeck
from eesizer_core.strategies import PatchLoopStrategy
from eesizer_core.contracts.artifacts import MetricsBundle, MetricValue


class FakeDeckBuildOperator:
    name = "fake_deck_build"
    version = "0.0.0"

    def run(self, inputs, ctx):
        kind = inputs.get("sim_kind", SimKind.ac)
        deck = SpiceDeck(text="", kind=kind, expected_outputs={})
        return OperatorResult(outputs={"deck": deck})


class FakeSimRunOperator:
    name = "fake_sim_run"
    version = "0.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, inputs, ctx):
        self.calls += 1
        if self.calls == 1:
            raise SimulationError("baseline sim failed")
        deck = inputs["deck"]
        stage = inputs.get("stage", deck.kind.value)
        stage_dir = Path(ctx.run_dir()) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_dir / f"ngspice_{deck.kind.value}.log"
        log_path.write_text("ok\n", encoding="utf-8")
        raw = RawSimData(
            kind=deck.kind,
            run_dir=stage_dir,
            outputs={},
            log_path=log_path,
            cmdline=["fake"],
            returncode=0,
        )
        return OperatorResult(outputs={"raw_data": raw})


class FakeMetricsOperator:
    name = "fake_metrics"
    version = "0.0.0"

    def run(self, inputs, ctx):
        metrics = MetricsBundle(
            values={"ac_mag_db_at_1k": MetricValue(name="ac_mag_db_at_1k", value=-10.0, unit="dB")}
        )
        return OperatorResult(outputs={"metrics": metrics})


def test_strategy_baseline_sim_error_is_recorded_and_retried(tmp_path):
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    policy = FixedSequencePolicy(patches=[])

    class DummyCtx:
        def run_dir(self):
            return tmp_path

    strategy = PatchLoopStrategy(
        policy=policy,
        deck_build_op=FakeDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
        max_patch_retries=1,
    )
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))

    result = strategy.run(spec=spec, source=source, ctx=DummyCtx(), cfg=cfg)

    assert result.history
    baseline = result.history[0]
    assert "attempts" in baseline
    assert any(
        attempt["guard"]
        and any(
            "baseline sim failed" in reason
            for check in attempt["guard"]["checks"]
            for reason in check["reasons"]
        )
        for attempt in baseline["attempts"]
    )
    assert result.stop_reason in {StopReason.policy_stop, StopReason.reached_target, StopReason.no_improvement}
