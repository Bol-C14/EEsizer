import pytest

from eesizer_core.contracts.errors import ValidationError
from eesizer_core.operators.llm import LLMCallOperator, LLMConfig, LLMRequest
from eesizer_core.runtime.context import RunContext


def _request() -> LLMRequest:
    return LLMRequest(system="sys", user="hello", config=LLMConfig(provider="mock", model="mock"))


def test_llm_call_rejects_absolute_stage(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    op = LLMCallOperator()

    with pytest.raises(ValidationError, match="relative"):
        op.run({"request": _request(), "stage": "/tmp/llm", "mock_response": "{}"}, ctx)


def test_llm_call_rejects_parent_traversal_stage(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    op = LLMCallOperator()

    with pytest.raises(ValidationError, match="must not contain"):
        op.run({"request": _request(), "stage": "llm/../evil", "mock_response": "{}"}, ctx)
