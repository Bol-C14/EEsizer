from __future__ import annotations

import pytest

from eesizer_core.contracts.plan_options import validate_llm_plan_options
from eesizer_core.runtime.plan_validator import validate_plan_options_semantics
from eesizer_core.runtime.session_plan_tools import build_session_plan_registry


def test_step8_plan_schema_validation_ok() -> None:
    payload = {
        "options": [
            {
                "title": "A",
                "plan": [
                    {
                        "id": "a01",
                        "op": "run_grid_search",
                        "inputs": ["session_run_dir", "source"],
                        "outputs": ["p1/run_dir", "p1/topk", "p1/pareto"],
                        "params": {},
                    }
                ],
            }
        ]
    }
    parsed = validate_llm_plan_options(payload)
    reg = build_session_plan_registry()
    report = validate_plan_options_semantics(parsed, registry=reg)
    assert report.ok is True


def test_step8_plan_schema_validation_rejects_duplicate_action_ids() -> None:
    payload = {
        "options": [
            {
                "title": "A",
                "plan": [
                    {"id": "x", "op": "run_grid_search", "inputs": [], "outputs": [], "params": {}},
                    {"id": "x", "op": "run_grid_search", "inputs": [], "outputs": [], "params": {}},
                ],
            }
        ]
    }
    with pytest.raises(Exception):
        validate_llm_plan_options(payload)

