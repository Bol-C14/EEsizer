from __future__ import annotations

"""LLM-backed policy that returns structured Patch JSON only."""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional
import json

from ..contracts import Patch, ParamSpace
from ..contracts.policy import Observation, Policy
from ..contracts.enums import PatchOpType
from .llm_patch_parser import PatchParseError, parse_patch_json


_SYSTEM_PROMPT = (
    "You are a circuit sizing assistant. "
    "Return ONLY a JSON object that conforms to patch.schema.json. "
    "Your response must be valid JSON ONLY. Do not include explanations outside JSON. "
    "Do not output netlists or explanations. "
    "Prefer 1-3 patch ops per response."
)


def _fmt_objectives(spec) -> str:
    lines = []
    for obj in spec.objectives:
        target = "None" if obj.target is None else obj.target
        tol = "None" if obj.tol is None else obj.tol
        lines.append(
            f"- metric={obj.metric} sense={obj.sense} target={target} tol={tol} weight={obj.weight}"
        )
    return "\n".join(lines) if lines else "- (none)"


def _fmt_metrics(metrics) -> str:
    if not metrics.values:
        return "- (none)"
    lines = []
    for name, mv in metrics.values.items():
        unit = mv.unit or ""
        passed = mv.passed
        lines.append(f"- {name}: value={mv.value} unit={unit} passed={passed}")
    return "\n".join(lines)


def _fmt_param_space(param_space: ParamSpace, param_values: Mapping[str, Any]) -> str:
    lines = []
    for param in param_space.params:
        if param.frozen:
            continue
        current = param_values.get(param.param_id, "N/A")
        lower = "None" if param.lower is None else param.lower
        upper = "None" if param.upper is None else param.upper
        lines.append(f"- {param.param_id}: current={current} bounds=[{lower}, {upper}] unit={param.unit}")
    return "\n".join(lines) if lines else "- (none)"


def _fmt_guard_feedback(notes: Mapping[str, Any]) -> str:
    report = notes.get("last_guard_report")
    failures = notes.get("last_guard_failures")
    if report:
        return json.dumps(report, indent=2, sort_keys=True)
    if failures:
        return json.dumps(failures, indent=2, sort_keys=True)
    return "(none)"


@dataclass
class LLMPatchPolicy(Policy):
    """LLM policy that emits Patch JSON and retries on parse errors."""

    name: str = "llm_patch"
    version: str = "0.1.0"

    provider: str = "mock"
    model: str = "gpt-4.1"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    response_schema_name: str = "patch.schema.json"

    max_retries: int = 2
    system_prompt: str = _SYSTEM_PROMPT

    llm_operator: Any = None
    mock_responses: list[str] = field(default_factory=list)
    mock_fn: Optional[Callable[[str], str]] = None

    _mock_index: int = field(default=0, init=False, repr=False)
    _manifest_updated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.llm_operator is None:
            from ..operators.llm import LLMCallOperator

            self.llm_operator = LLMCallOperator()

    def _update_manifest(self, ctx: Any) -> None:
        if self._manifest_updated:
            return
        if ctx is None or not hasattr(ctx, "manifest"):
            return
        try:
            manifest = ctx.manifest()
        except Exception:
            return
        manifest.environment.setdefault("llm_provider", self.provider)
        manifest.environment.setdefault("llm_model", self.model)
        manifest.environment.setdefault("llm_temperature", self.temperature)
        if self.max_tokens is not None:
            manifest.environment.setdefault("llm_max_tokens", self.max_tokens)
        if self.seed is not None:
            manifest.environment.setdefault("llm_seed", self.seed)
        manifest.environment.setdefault("llm_response_schema", self.response_schema_name)
        self._manifest_updated = True

    def _allowed_params(self, param_space: ParamSpace) -> set[str]:
        return {p.param_id for p in param_space.params if not p.frozen}

    def _build_user_prompt(self, obs: Observation, param_values: Mapping[str, Any]) -> str:
        objectives = _fmt_objectives(obs.spec)
        metrics = _fmt_metrics(obs.metrics)
        params = _fmt_param_space(obs.param_space, param_values)
        current_score = obs.notes.get("current_score")
        best_score = obs.notes.get("best_score")
        guard_feedback = _fmt_guard_feedback(obs.notes)
        attempt = obs.notes.get("attempt", 0)
        return (
            "Optimization objective:\n"
            "- Score is a penalty to MINIMIZE (lower is better).\n"
            "- score = 0.0 means ALL objectives are satisfied.\n"
            "- Propose a patch that reduces the score and improves objective margins.\n\n"
            "You must return a JSON object with keys: patch, stop, notes. "
            "Each patch item must be {param, op, value, why?}, where op in "
            f"{[t.value for t in PatchOpType]}.\n\n"
            f"Iteration: {obs.iteration}\n"
            f"Attempt: {attempt}\n"
            f"Current score: {current_score}\n"
            f"Best score: {best_score}\n\n"
            "When proposing changes, prioritize satisfying constraints; then improve margins.\n\n"
            "Objectives:\n"
            f"{objectives}\n\n"
            "Metrics:\n"
            f"{metrics}\n\n"
            "Allowed parameters (non-frozen):\n"
            f"{params}\n\n"
            "Last guard feedback (if any):\n"
            f"{guard_feedback}\n"
        )

    def _stage_name(self, obs: Observation, retry_idx: int) -> str:
        attempt = obs.notes.get("attempt", 0)
        base = f"llm/llm_i{obs.iteration:03d}_a{attempt:02d}"
        if retry_idx > 0:
            return f"{base}_r{retry_idx:02d}"
        return base

    def _mock_factor_for_score(self, score: float) -> float:
        if score > 0.5:
            return 0.8
        if score > 0.1:
            return 0.9
        return 0.97

    def _mock_response_for_call(
        self,
        request_text: str,
        obs: Observation,
        param_values: Mapping[str, Any],
        allowed_params: set[str],
    ) -> str:
        if self.mock_fn is not None:
            return str(self.mock_fn(request_text))
        if not self.mock_responses:
            return self._mock_response_from_obs(obs, param_values, allowed_params)
        if self._mock_index >= len(self.mock_responses):
            response = self.mock_responses[-1]
        else:
            response = self.mock_responses[self._mock_index]
        self._mock_index += 1
        return response

    def _mock_response_from_obs(
        self,
        obs: Observation,
        param_values: Mapping[str, Any],
        allowed_params: set[str],
    ) -> str:
        candidates = sorted(allowed_params)
        if not candidates:
            return '{"patch": [], "notes": "mock_no_params"}'

        current_score = obs.notes.get("current_score")
        score = float(current_score) if isinstance(current_score, (int, float)) else 1.0
        factor = self._mock_factor_for_score(score)
        abs_min = 1e-12
        abs_max = 1e12

        for _ in range(len(candidates)):
            param_id = candidates[self._mock_index % len(candidates)]
            self._mock_index += 1
            current = param_values.get(param_id)
            if not isinstance(current, (int, float)):
                continue

            candidate = float(current) * factor
            patch_op = "mul"
            patch_value: float = factor
            why = "mock_step"
            param_def = obs.param_space.get(param_id)
            lower = param_def.lower if param_def is not None else None
            upper = param_def.upper if param_def is not None else None

            if lower is not None and candidate < lower:
                candidate = lower
                patch_op = "set"
                patch_value = float(lower)
                why = "mock_clamp"
            if upper is not None and candidate > upper:
                candidate = upper
                patch_op = "set"
                patch_value = float(upper)
                why = "mock_clamp"

            if lower is None and upper is None:
                if abs(candidate) < abs_min:
                    candidate = abs_min if candidate >= 0 else -abs_min
                    patch_op = "set"
                    patch_value = float(candidate)
                    why = "mock_clamp"
                if abs(candidate) > abs_max:
                    candidate = abs_max if candidate >= 0 else -abs_max
                    patch_op = "set"
                    patch_value = float(candidate)
                    why = "mock_clamp"

            if abs(candidate - float(current)) <= 1e-12:
                continue

            payload = {
                "patch": [{"param": param_id, "op": patch_op, "value": patch_value, "why": why}],
                "notes": "mock_response",
            }
            return json.dumps(payload)

        return '{"patch": [], "notes": "mock_no_change"}'

    def _write_parse_artifact(self, ctx: Any, stage: str, filename: str, payload: str) -> None:
        if ctx is None or not hasattr(ctx, "run_dir"):
            return
        stage_dir = ctx.run_dir() / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / filename).write_text(payload, encoding="utf-8")

    def propose(self, obs: Observation, ctx: Any) -> Patch:  # type: ignore[override]
        self._update_manifest(ctx)
        allowed_params = self._allowed_params(obs.param_space)
        if not allowed_params:
            return Patch(stop=True, notes="no_tunable_params")

        param_values = obs.notes.get("param_values")
        if not isinstance(param_values, Mapping) or not param_values:
            return Patch(stop=True, notes="missing_param_values")
        current_score = obs.notes.get("current_score")
        if not isinstance(current_score, (int, float)):
            return Patch(stop=True, notes="missing_current_score")

        base_prompt = self._build_user_prompt(obs, param_values)
        last_error: Optional[str] = None

        for retry in range(self.max_retries + 1):
            prompt = base_prompt
            if last_error:
                prompt += (
                    "\n\nYour last output did not match the schema. "
                    f"Error: {last_error}. Return only valid JSON."
                )

            stage = self._stage_name(obs, retry)
            request_payload = {
                "system": self.system_prompt,
                "user": prompt,
                "config": {
                    "provider": self.provider,
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "seed": self.seed,
                },
                "response_schema_name": self.response_schema_name,
            }

            inputs = {"request": request_payload, "stage": stage}
            if self.provider == "mock":
                inputs["mock_response"] = self._mock_response_for_call(prompt, obs, param_values, allowed_params)
            try:
                llm_result = self.llm_operator.run(inputs, ctx)
            except Exception as exc:
                self._write_parse_artifact(ctx, stage, "call_error.txt", str(exc))
                return Patch(stop=True, notes="llm_call_failed")
            response_text = llm_result.outputs.get("response_text", "")

            try:
                patch = parse_patch_json(response_text, allowed_params=allowed_params)
            except PatchParseError as exc:
                last_error = str(exc)
                self._write_parse_artifact(ctx, stage, "parse_error.txt", last_error)
                continue

            self._write_parse_artifact(ctx, stage, "parsed_patch.json", json.dumps(
                {
                    "patch": [
                        {"param": op.param, "op": op.op.value, "value": op.value, "why": op.why}
                        for op in patch.ops
                    ],
                    "stop": patch.stop,
                    "notes": patch.notes,
                },
                indent=2,
                sort_keys=True,
            ))
            return patch

        return Patch(stop=True, notes="llm_parse_failed")
