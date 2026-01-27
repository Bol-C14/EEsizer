from __future__ import annotations

from pathlib import Path
import json

from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.session_llm_advice import SessionLLMAdviseOperator
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.session_store import SessionStore


def test_step7_llm_advise_repairs_invalid_json(tmp_path: Path) -> None:
    # Minimal session + grid checkpoint so context builder has something to read.
    run_dir = tmp_path / "runs" / "sess1"
    store = SessionStore(run_dir)
    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3), seed=0, notes={})
    store.create_session(session_id="sess1", bench_id="ota", seed=0, spec=spec, cfg=cfg)

    (run_dir / "session" / "meta_report.md").write_text("# Session Meta Report\n", encoding="utf-8")
    grid_run_dir = tmp_path / "runs" / "grid1"
    (grid_run_dir / "search").mkdir(parents=True, exist_ok=True)
    (grid_run_dir / "report.md").write_text("## Run Summary\nok\n", encoding="utf-8")
    (grid_run_dir / "search" / "topk.json").write_text("[]", encoding="utf-8")
    (grid_run_dir / "search" / "pareto.json").write_text("[]", encoding="utf-8")
    store.write_checkpoint("p1_grid", {"phase_id": "p1_grid", "input_hash": "sha256:stub", "run_dir": str(grid_run_dir)})

    # First response is invalid JSON; repair provides a valid insights object.
    bad = "{ this is not json"
    repaired_insights = json.dumps(
        {
            "summary": "ok",
            "tradeoffs": [],
            "sensitivity_rank": [],
            "robustness_notes": [],
            "recommended_actions": [],
        }
    )
    plan = json.dumps(
        {
            "options": [
                {
                    "title": "Plan A",
                    "intent": "mock",
                    "spec_delta": None,
                    "cfg_delta": {"notes": {"grid_search": {"levels": 10}}},
                    "plan": None,
                    "expected_effects": ["mock"],
                    "risks": ["mock"],
                    "budget_estimate": {"iters": 10, "corners": 0},
                }
            ]
        }
    )

    ctx = RunContext(workspace_root=tmp_path, run_id="sess1", seed=0)
    op = SessionLLMAdviseOperator()
    res = op.run(
        {
            "session_run_dir": run_dir,
            "llm_config": {"provider": "mock", "model": "mock", "temperature": 0.0},
            "mock_response": bad,
            "mock_repairs": [repaired_insights],
            "mock_plan": plan,
            "max_repairs": 1,
        },
        ctx=ctx,
    )

    assert res.outputs["advice_rev"] == 0
    advice_rel = res.outputs["advice_dir"]
    assert (run_dir / advice_rel / "insights.json").exists()
    assert (run_dir / advice_rel / "proposal.json").exists()
    assert (run_dir / "session" / "llm" / "narrative.md").exists()

