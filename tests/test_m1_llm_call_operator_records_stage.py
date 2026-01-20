import json

from eesizer_core.operators.llm import LLMCallOperator, LLMConfig, LLMRequest
from eesizer_core.runtime.context import RunContext


def test_llm_call_operator_records_stage(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    op = LLMCallOperator()
    request = LLMRequest(system="sys", user="hello", config=LLMConfig(provider="mock", model="mock"))
    op.run(
        {"request": request, "stage": "llm/llm_i001_a00", "mock_response": '{"patch": []}'},
        ctx,
    )

    stage_dir = ctx.run_dir() / "llm" / "llm_i001_a00"
    assert (stage_dir / "request.json").exists()
    assert (stage_dir / "prompt.txt").exists()
    assert (stage_dir / "response.txt").exists()

    calls_path = ctx.run_dir() / "provenance" / "operator_calls.jsonl"
    lines = calls_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert any(p.get("operator") == "llm_call" for p in payloads)
