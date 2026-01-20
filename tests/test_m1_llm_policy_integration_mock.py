from pathlib import Path

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import LLMPatchPolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import NoOptBaselineStrategy, PatchLoopStrategy


@pytest.mark.integration
def test_llm_patch_policy_mock_improves_ac_mag(tmp_path):
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    netlist_path = examples_dir / "circuits" / "rc_lowpass.sp"
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    baseline_ctx = RunContext(workspace_root=tmp_path / "baseline")
    baseline_cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    baseline = NoOptBaselineStrategy()
    baseline_result = baseline.run(spec=spec, source=source, ctx=baseline_ctx, cfg=baseline_cfg)
    baseline_mv = baseline_result.best_metrics.get("ac_mag_db_at_1k")
    assert baseline_mv is not None

    policy = LLMPatchPolicy(
        provider="mock",
        mock_responses=['{"patch":[{"param":"r1.value","op":"mul","value":0.5,"why":"mock"}]}'],
        max_retries=0,
    )
    strategy = PatchLoopStrategy(policy=policy)
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=6, no_improve_patience=3),
        notes={"param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]}},
    )
    ctx = RunContext(workspace_root=tmp_path / "opt")

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    best_mv = result.best_metrics.get("ac_mag_db_at_1k")
    assert best_mv is not None

    assert best_mv.value is not None
    assert baseline_mv.value is not None
    assert best_mv.value >= baseline_mv.value + 0.2
    assert ctx.run_dir().exists()
