from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ...contracts import Patch
from ...contracts.policy import Observation


def _llm_patch_payload(patch: Patch) -> dict[str, Any]:
    return {
        "patch": [
            {"param": op.param, "op": getattr(op.op, "value", op.op), "value": op.value, "why": getattr(op, "why", "")}
            for op in patch.ops
        ],
        "stop": patch.stop,
        "notes": patch.notes,
    }


def _llm_stage_name(obs: Observation, retry_idx: int) -> str:
    attempt = obs.notes.get("attempt", 0)
    base = f"llm/llm_i{obs.iteration:03d}_a{attempt:02d}"
    if retry_idx > 0:
        return f"{base}_r{retry_idx:02d}"
    return base


def _write_llm_artifact(ctx: Any, stage: str, filename: str, payload: str) -> None:
    if ctx is None or not hasattr(ctx, "run_dir"):
        return
    stage_dir = Path(ctx.run_dir()) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / filename).write_text(payload, encoding="utf-8")


def is_llm_policy(policy: Any) -> bool:
    return hasattr(policy, "build_request") and hasattr(policy, "parse_response")


def _propose_llm_patch(
    policy: Any,
    llm_call_op: Any,
    obs: Observation,
    ctx: Any,
) -> Patch:
    last_error: str | None = None
    max_retries = int(getattr(policy, "max_retries", 0))

    for retry in range(max_retries + 1):
        request_payload, stop_reason = policy.build_request(obs, last_error=last_error)
        if stop_reason:
            return Patch(stop=True, notes=stop_reason)
        if not isinstance(request_payload, dict):
            return Patch(stop=True, notes="llm_request_missing")

        stage = _llm_stage_name(obs, retry)
        inputs = {"request": request_payload, "stage": stage}
        provider = request_payload.get("config", {}).get("provider", getattr(policy, "provider", None))
        if provider == "mock" and hasattr(policy, "mock_response"):
            prompt = request_payload.get("user", "")
            inputs["mock_response"] = policy.mock_response(prompt, obs)
        try:
            llm_result = llm_call_op.run(inputs, ctx)
        except Exception as exc:
            _write_llm_artifact(ctx, stage, "call_error.txt", str(exc))
            return Patch(stop=True, notes="llm_call_failed")

        response_text = llm_result.outputs.get("response_text", "")
        try:
            patch = policy.parse_response(response_text, obs)
        except Exception as exc:
            last_error = str(exc)
            _write_llm_artifact(ctx, stage, "parse_error.txt", last_error)
            continue

        _write_llm_artifact(
            ctx,
            stage,
            "parsed_patch.json",
            json.dumps(_llm_patch_payload(patch), indent=2, sort_keys=True),
        )
        return patch

    return Patch(stop=True, notes="llm_parse_failed")


def propose_patch(policy: Any, llm_call_op: Any, obs: Observation, ctx: Any) -> Patch:
    if is_llm_policy(policy):
        return _propose_llm_patch(policy, llm_call_op, obs, ctx)
    return policy.propose(obs, ctx)
