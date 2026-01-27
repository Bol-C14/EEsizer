from __future__ import annotations

from pathlib import Path
import json

from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.session_llm_advice import SessionLLMAdviseOperator
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.session_store import SessionStore


def test_step7_reject_does_not_change_spec_or_cfg_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sess1"
    store = SessionStore(run_dir)
    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3), seed=0, notes={})
    store.create_session(session_id="sess1", bench_id="ota", seed=0, spec=spec, cfg=cfg)
    (run_dir / "session" / "meta_report.md").write_text("# Session Meta Report\n", encoding="utf-8")

    # Create the minimum p1 checkpoint required by the advice context builder.
    grid_run_dir = tmp_path / "runs" / "grid1"
    (grid_run_dir / "search").mkdir(parents=True, exist_ok=True)
    (grid_run_dir / "report.md").write_text("## Run Summary\nok\n", encoding="utf-8")
    (grid_run_dir / "search" / "topk.json").write_text("[]", encoding="utf-8")
    (grid_run_dir / "search" / "pareto.json").write_text("[]", encoding="utf-8")
    store.write_checkpoint("p1_grid", {"phase_id": "p1_grid", "input_hash": "sha256:stub", "run_dir": str(grid_run_dir)})

    ctx = RunContext(workspace_root=tmp_path, run_id="sess1", seed=0)
    op = SessionLLMAdviseOperator()
    op.run(
        {
            "session_run_dir": run_dir,
            "llm_config": {"provider": "mock", "model": "mock", "temperature": 0.0},
            "mock_response": json.dumps(
                {
                    "summary": "ok",
                    "tradeoffs": [],
                    "sensitivity_rank": [],
                    "robustness_notes": [],
                    "recommended_actions": [],
                }
            ),
            "mock_plan": json.dumps(
                {
                    "options": [
                        {
                            "title": "No-op",
                            "intent": "mock",
                            "spec_delta": None,
                            "cfg_delta": None,
                            "plan": None,
                            "expected_effects": [],
                            "risks": [],
                            "budget_estimate": {"iters": 0, "corners": 0},
                        }
                    ]
                }
            ),
        },
        ctx=ctx,
    )

    state_before = store.load_session_state()
    assert state_before.latest_advice_rev == 0
    spec_hash_before = state_before.spec_hash
    cfg_hash_before = state_before.cfg_hash
    spec_rev_before = state_before.spec_rev
    cfg_rev_before = state_before.cfg_rev

    store.mark_advice_decision(rev=0, decision="rejected", reason="unit_test_reject")

    state_after = store.load_session_state()
    assert state_after.spec_hash == spec_hash_before
    assert state_after.cfg_hash == cfg_hash_before
    assert state_after.spec_rev == spec_rev_before
    assert state_after.cfg_rev == cfg_rev_before

