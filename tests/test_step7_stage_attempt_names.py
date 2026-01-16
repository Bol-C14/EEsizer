from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective, MetricsBundle, MetricValue
from eesizer_core.contracts.enums import SimKind, SourceKind
from eesizer_core.contracts.guards import GuardCheck
from eesizer_core.contracts.operators import OperatorResult
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.artifacts import RawSimData, SpiceDeck
from eesizer_core.strategies import PatchLoopStrategy
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType


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

    def run(self, inputs, ctx):
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


class FailingOnceBehaviorGuard:
    name = "behavior_guard"
    version = "0.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, inputs, ctx):
        self.calls += 1
        ok = self.calls != 2
        reasons = () if ok else ("forced_fail",)
        check = GuardCheck(name="behavior_guard", ok=ok, severity="hard", reasons=reasons)
        return OperatorResult(outputs={"check": check})


def test_stage_names_include_attempt_suffix(tmp_path):
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=0.0, sense="ge"),))

    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    policy = FixedSequencePolicy(patches=[patch, patch])

    strategy = PatchLoopStrategy(
        policy=policy,
        deck_build_op=FakeDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
        behavior_guard_op=FailingOnceBehaviorGuard(),
        max_patch_retries=1,
    )
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert (run_dir / "ac_i001_a00").exists()
    assert (run_dir / "ac_i001_a01").exists()
