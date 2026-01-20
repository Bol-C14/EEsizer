from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind, StopReason
from eesizer_core.contracts.errors import ValidationError
from eesizer_core.contracts.operators import OperatorResult
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim import DeckBuildOperator
from eesizer_core.sim.artifacts import RawSimData
from eesizer_core.strategies import PatchLoopStrategy


class RecordingDeckBuildOperator:
    name = "recording_deck_build"
    version = "0.0.0"

    def __init__(self) -> None:
        self.seen_texts: list[str] = []
        self._op = DeckBuildOperator()

    def run(self, inputs, ctx):
        src = inputs.get("circuit_source")
        if src is not None:
            self.seen_texts.append(src.text)
        return self._op.run(inputs, ctx)


class FakeSimRunOperator:
    name = "fake_sim_run"
    version = "0.0.0"

    def run(self, inputs, ctx):
        deck = inputs["deck"]
        stage = inputs.get("stage", deck.kind.value)
        stage_dir = Path(ctx.run_dir()) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_dir / "ngspice.log"
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
            values={"ac_mag_db_at_1k": MetricValue(name="ac_mag_db_at_1k", value=-3.0, unit="dB")}
        )
        return OperatorResult(outputs={"metrics": metrics})


class FailingDeckBuildOperator:
    name = "failing_deck_build"
    version = "0.0.0"

    def run(self, inputs, ctx):
        raise ValidationError("deck build failed")


def _strategy_cfg() -> StrategyConfig:
    return StrategyConfig(budget=OptimizationBudget(max_iterations=0, no_improve_patience=1))


def test_patch_loop_baseline_uses_sanitized_netlist(tmp_path):
    netlist = "\n".join(
        [
            "* baseline sanitize check",
            ".include /abs/evil.sp",
            "R1 in out 1k",
            ".control",
            "echo hello",
            ".endc",
            ".end",
        ]
    )
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-1.0, sense="ge"),))

    deck_op = RecordingDeckBuildOperator()
    strategy = PatchLoopStrategy(
        policy=FixedSequencePolicy(),
        deck_build_op=deck_op,
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
        max_patch_retries=0,
    )
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=_strategy_cfg())

    assert deck_op.seen_texts
    seen = deck_op.seen_texts[0]
    assert ".control" not in seen.lower()
    assert "/abs/evil.sp" not in seen


def test_baseline_validation_error_becomes_guard_fail(tmp_path):
    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n")
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-1.0, sense="ge"),))

    strategy = PatchLoopStrategy(
        policy=FixedSequencePolicy(),
        deck_build_op=FailingDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
        max_patch_retries=0,
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=_strategy_cfg())

    assert result.stop_reason == StopReason.guard_failed
    assert result.history
    assert result.history[0]["guard"]["ok"] is False
