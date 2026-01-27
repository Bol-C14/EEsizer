from __future__ import annotations

from eesizer_core.contracts.plan_options import validate_llm_plan_options
from eesizer_core.runtime.plan_validator import validate_plan_options_semantics
from eesizer_core.runtime.session_plan_tools import build_session_plan_registry


def test_step8_plan_semantic_rejects_levels_out_of_bounds() -> None:
    payload = {
        "options": [
            {
                "title": "A",
                "plan": [
                    {
                        "id": "a01",
                        "op": "update_cfg",
                        "inputs": ["session_run_dir", "source"],
                        "outputs": ["session/cfg_rev", "session/cfg_hash"],
                        "params": {"cfg_delta": {"notes": {"grid_search": {"levels": 999}}}},
                    }
                ],
            }
        ]
    }
    parsed = validate_llm_plan_options(payload)
    reg = build_session_plan_registry()
    report = validate_plan_options_semantics(parsed, registry=reg)
    assert report.ok is False
    assert any("grid_search.levels" in e for e in report.errors)

