"""Operator wrapper for LLM calls with audit-friendly artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import json
import os

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ...runtime.recorder import RunRecorder
from .contracts import LLMConfig, LLMRequest, LLMResponse


def _prompt_text(request: LLMRequest) -> str:
    return f"SYSTEM:\n{request.system}\n\nUSER:\n{request.user}\n"


def _request_payload(request: LLMRequest) -> dict[str, Any]:
    payload = {
        "system": request.system,
        "user": request.user,
        "config": asdict(request.config),
        "response_schema_name": request.response_schema_name,
    }
    return payload


def _resolve_stage_dir(ctx: Any, stage: str | None) -> Path:
    if ctx is None or not hasattr(ctx, "run_dir"):
        raise ValidationError("LLMCallOperator requires a RunContext with run_dir()")
    stage_name = stage or "llm"
    stage_path = Path(stage_name)
    if stage_path.is_absolute():
        raise ValidationError("LLM stage path must be relative")
    if any(part == ".." for part in stage_path.parts):
        raise ValidationError("LLM stage path must not contain '..'")
    stage_dir = Path(ctx.run_dir()) / stage_path
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _record_operator_call(recorder: RunRecorder, provenance: Provenance) -> None:
    payload = {
        "operator": provenance.operator,
        "version": provenance.version,
        "start_time": provenance.start_time,
        "end_time": provenance.end_time,
        "duration_s": provenance.duration_s(),
        "command": provenance.command,
        "inputs": {k: v.sha256 for k, v in provenance.inputs.items()},
        "outputs": {k: v.sha256 for k, v in provenance.outputs.items()},
        "notes": dict(provenance.notes),
    }
    recorder.append_jsonl("provenance/operator_calls.jsonl", payload)


class LLMCallOperator(Operator):
    """Call an LLM backend and persist request/response artifacts."""

    name = "llm_call"
    version = "0.1.0"

    def __init__(self) -> None:
        self._mock_index = 0

    def _coerce_request(self, payload: Any) -> LLMRequest:
        if isinstance(payload, LLMRequest):
            return payload
        if not isinstance(payload, Mapping):
            raise ValidationError("LLMCallOperator: 'request' must be LLMRequest or mapping")
        system = payload.get("system")
        user = payload.get("user")
        if not isinstance(system, str) or not isinstance(user, str):
            raise ValidationError("LLMCallOperator: request.system and request.user must be strings")
        cfg_payload = payload.get("config", {})
        if isinstance(cfg_payload, LLMConfig):
            config = cfg_payload
        elif isinstance(cfg_payload, Mapping):
            cfg_kwargs: dict[str, Any] = {}
            for key in ("provider", "model", "temperature", "max_tokens", "seed"):
                if key in cfg_payload:
                    cfg_kwargs[key] = cfg_payload[key]
            config = LLMConfig(**cfg_kwargs)
        else:
            raise ValidationError("LLMCallOperator: request.config must be LLMConfig or mapping")
        response_schema_name = payload.get("response_schema_name")
        if response_schema_name is not None and not isinstance(response_schema_name, str):
            raise ValidationError("LLMCallOperator: response_schema_name must be a string if provided")
        return LLMRequest(
            system=system,
            user=user,
            config=config,
            response_schema_name=response_schema_name,
        )

    def _call_openai(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValidationError("OpenAI backend requires 'openai'; install with pip install -e \".[llm]\"") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValidationError("OPENAI_API_KEY is not set; export it before using the OpenAI backend")

        client = OpenAI(api_key=api_key)
        params: dict[str, Any] = {
            "model": request.config.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.config.temperature,
        }
        if request.config.max_tokens is not None:
            params["max_tokens"] = request.config.max_tokens
        if request.config.seed is not None:
            params["seed"] = request.config.seed

        response = client.chat.completions.create(**params)
        text = ""
        if response and getattr(response, "choices", None):
            choice = response.choices[0]
            msg = getattr(choice, "message", None)
            text = getattr(msg, "content", "") or ""

        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif hasattr(usage_obj, "__dict__"):
            usage = dict(usage_obj.__dict__)
        else:
            usage = {
                key: getattr(usage_obj, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(usage_obj, key)
            }

        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()
        else:
            raw = {"text": text}

        return LLMResponse(text=text, usage=usage, raw=raw)

    def _call_mock(self, request: LLMRequest, inputs: Mapping[str, Any]) -> LLMResponse:
        mock_response = inputs.get("mock_response")
        if isinstance(mock_response, str):
            return LLMResponse(text=mock_response, usage={}, raw={})

        mock_fn = inputs.get("mock_fn")
        if callable(mock_fn):
            try:
                text = mock_fn(request)
            except TypeError:
                text = mock_fn(_prompt_text(request))
            return LLMResponse(text=str(text), usage={}, raw={})

        mock_responses = inputs.get("mock_responses")
        if isinstance(mock_responses, list) and mock_responses:
            if self._mock_index >= len(mock_responses):
                text = mock_responses[-1]
            else:
                text = mock_responses[self._mock_index]
            self._mock_index += 1
            return LLMResponse(text=str(text), usage={}, raw={})

        raise ValidationError("mock provider requires mock_response, mock_responses, or mock_fn")

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        request = self._coerce_request(inputs.get("request"))

        stage_dir = _resolve_stage_dir(ctx, inputs.get("stage"))
        prompt = _prompt_text(request)
        request_payload = _request_payload(request)

        (stage_dir / "request.json").write_text(
            json.dumps(request_payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        (stage_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        provider = request.config.provider
        if provider == "openai":
            response = self._call_openai(request)
        elif provider == "mock":
            response = self._call_mock(request, inputs)
        else:
            raise ValidationError(f"unsupported LLM provider: {provider}")

        (stage_dir / "response.txt").write_text(response.text, encoding="utf-8")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["request"] = ArtifactFingerprint(sha256=stable_hash_json(request_payload))
        provenance.inputs["prompt"] = ArtifactFingerprint(sha256=stable_hash_str(prompt))
        provenance.outputs["response_text"] = ArtifactFingerprint(sha256=stable_hash_str(response.text))
        provenance.outputs["response_raw"] = ArtifactFingerprint(sha256=stable_hash_json(response.raw))
        provenance.notes.update(
            {
                "provider": request.config.provider,
                "model": request.config.model,
                "temperature": request.config.temperature,
                "max_tokens": request.config.max_tokens,
                "seed": request.config.seed,
                "response_schema_name": request.response_schema_name,
                "stage_dir": str(stage_dir),
                "usage": dict(response.usage),
            }
        )
        provenance.finish()

        if hasattr(ctx, "recorder"):
            try:
                recorder = ctx.recorder()
                _record_operator_call(recorder, provenance)
            except Exception:
                pass

        return OperatorResult(
            outputs={
                "response_text": response.text,
                "response_usage": dict(response.usage),
                "response_raw": dict(response.raw),
                "stage_dir": stage_dir,
            },
            provenance=provenance,
        )
