from __future__ import annotations

from eesizer_core.operators.llm_prompt import build_report_interpret_prompt, build_tool_plan_prompt


def test_step7_prompt_builder_includes_schema_and_hard_rules() -> None:
    context = {
        "session": {"session_id": "s", "bench_id": "ota"},
        "spec": {"objectives": []},
        "cfg": {"budget": {"max_iterations": 10}, "notes": {}},
        "evidence": ["search/topk.json"],
    }

    sys_a, user_a = build_report_interpret_prompt(context)
    assert "Hard rules" in sys_a
    assert "Output MUST be strict JSON" in sys_a
    assert "matches this schema" in user_a
    assert "\"summary\"" in user_a

    insights = {
        "summary": "ok",
        "tradeoffs": [],
        "sensitivity_rank": [],
        "robustness_notes": [],
        "recommended_actions": [],
    }
    sys_b, user_b = build_tool_plan_prompt(insights=insights, context=context)
    assert "Hard rules" in sys_b
    assert "spec_delta" in sys_b
    assert "cfg_delta" in sys_b
    assert "Return a JSON object that matches this schema" in user_b
