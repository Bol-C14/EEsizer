from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, Objective, Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType, SimKind, SourceKind, StopReason
from eesizer_core.contracts.errors import SimulationError
from eesizer_core.contracts.operators import OperatorResult
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import FixedSequencePolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.artifacts import RawSimData, SpiceDeck
from eesizer_core.strategies import PatchLoopStrategy
from eesizer_core.strategies.patch_loop.evaluate import measure_metrics


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

    def __init__(self, fail_kind: SimKind | None = None) -> None:
        self.fail_kind = fail_kind
        self.calls = 0

    def run(self, inputs, ctx):
        deck = inputs["deck"]
        self.calls += 1
        if self.fail_kind == deck.kind:
            raise SimulationError("sim failed")
        stage = inputs.get("stage", deck.kind.value)
        run_dir = Path(ctx.run_dir()) / stage if ctx is not None else Path(stage)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"ngspice_{deck.kind.value}.log"
        log_path.write_text("ok\n", encoding="utf-8")
        raw = RawSimData(
            kind=deck.kind,
            run_dir=run_dir,
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
        metrics = MetricsBundle()
        for name in inputs.get("metric_names", []) or []:
            metrics.values[str(name)] = MetricValue(name=str(name), value=0.0, unit="")
        return OperatorResult(outputs={"metrics": metrics})


def test_measure_metrics_counts_each_kind(tmp_path):
    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n.end\n")
    ctx = RunContext(workspace_root=tmp_path)
    result = measure_metrics(
        source=source,
        metric_groups={SimKind.ac: ["ac_mag_db_at_1k"], SimKind.tran: ["tran_rise_time"]},
        ctx=ctx,
        iter_idx=0,
        attempt_idx=0,
        recorder=None,
        manifest=None,
        measure_fn=None,
        deck_build_op=FakeDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(fail_kind=SimKind.tran),
        metrics_op=FakeMetricsOperator(),
    )

    assert result.sim_runs == 2
    assert result.sim_runs_ok == 1
    assert result.sim_runs_failed == 1
    assert not result.ok


def test_budget_stops_after_multi_kind_baseline(tmp_path):
    netlist = "R1 in out 1k\nC1 out 0 1p\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(
        objectives=(
            Objective(metric="ac_mag_db_at_1k", target=1.0, sense="ge"),
            Objective(metric="tran_rise_time", target=1.0, sense="le"),
        )
    )
    policy = FixedSequencePolicy(patches=[Patch(ops=(PatchOp(param="r1.value", op=PatchOpType.set, value=2.0),))])
    strategy = PatchLoopStrategy(
        policy=policy,
        deck_build_op=FakeDeckBuildOperator(),
        sim_run_op=FakeSimRunOperator(),
        metrics_op=FakeMetricsOperator(),
        max_patch_retries=0,
    )
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=1, max_sim_runs=1, no_improve_patience=1),
        notes={"guard_cfg": {"scan_logs": False}},
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    assert result.stop_reason is StopReason.budget_exhausted
