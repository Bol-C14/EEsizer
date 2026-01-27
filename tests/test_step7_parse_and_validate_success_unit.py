from __future__ import annotations

import json

from eesizer_core.contracts.llm_insights import validate_llm_insights
from eesizer_core.contracts.llm_proposal import validate_llm_proposal
from eesizer_core.operators.llm_parse import ParseJSONWithSchemaOperator


def test_step7_parse_and_validate_success() -> None:
    op = ParseJSONWithSchemaOperator()

    insights = {
        "summary": "ok",
        "tradeoffs": [{"x": "power_w", "y": "ugbw_hz", "pattern": "none", "evidence": ["search/topk.json"]}],
        "sensitivity_rank": [{"param_id": "m1.w", "impact": "high", "evidence": ["insights/sensitivity.json"]}],
        "robustness_notes": [{"worst_corner_id": "all_low", "why": "mock", "evidence": ["search/robust_topk.json"]}],
        "recommended_actions": [{"action_type": "increase_levels", "rationale": "mock"}],
    }
    out = op.run({"text": json.dumps(insights), "schema_name": "llm_insights", "validate_fn": validate_llm_insights}, ctx=None).outputs
    assert out["parsed"] is not None
    assert out["validation"]["ok"] is True

    proposal = {
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
    out2 = op.run(
        {"text": json.dumps(proposal), "schema_name": "llm_proposal", "validate_fn": validate_llm_proposal},
        ctx=None,
    ).outputs
    assert out2["parsed"] is not None
    assert out2["validation"]["ok"] is True

