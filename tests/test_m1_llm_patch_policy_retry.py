from pathlib import Path
import json

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.contracts.enums import SimKind, SourceKind
from eesizer_core.contracts.operators import OperatorResult
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import LLMPatchPolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.artifacts import RawSimData, SpiceDeck
from eesizer_core.strategies import PatchLoopStrategy


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


def test_llm_patch_policy_retries_on_parse_error(tmp_path):
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-5.0, sense="ge"),))

    policy = LLMPatchPolicy(
        provider="mock",
        mock_responses=[
            "not json",
            '{"patch":[{"param":"m1.w","op":"mul","value":1.1}]}',
        ],
        max_retries=1,
    )

    strategy = PatchLoopStrategy(
        policy=policy,
        deck_build_op=FakeDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
    )
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    ctx = RunContext(workspace_root=tmp_path)

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    stage_dir = ctx.run_dir() / "llm" / "llm_i001_a00"
    retry_dir = ctx.run_dir() / "llm" / "llm_i001_a00_r01"
    assert (stage_dir / "parse_error.txt").exists()
    assert (retry_dir / "parsed_patch.json").exists()

    calls_path = ctx.run_dir() / "provenance" / "operator_calls.jsonl"
    lines = calls_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    llm_calls = [p for p in payloads if p.get("operator") == "llm_call"]
    assert len(llm_calls) >= 2
