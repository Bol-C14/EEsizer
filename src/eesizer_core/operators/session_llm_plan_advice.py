from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json

from ..analysis.session_report import build_meta_report
from ..contracts.errors import ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.plan_options import PLAN_OPTIONS_SCHEMA, validate_llm_plan_options
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ..operators.llm.llm_call import LLMCallOperator
from ..operators.llm.contracts import LLMConfig, LLMRequest
from ..operators.llm_context.build_planning_context import build_session_llm_planning_context
from ..operators.llm_parse import ParseJSONWithSchemaOperator, build_repair_prompt
from ..operators.llm_prompt.build_plan_options_prompt import build_plan_options_prompt
from ..runtime.plan_validator import raise_on_invalid_plan, validate_plan_options_semantics
from ..runtime.session_plan_tools import build_session_plan_registry
from ..runtime.session_store import SessionStore
from ..runtime.tool_catalog import build_tool_catalog


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionLLMPlanAdviceOperator(Operator):
    """Generate PlanOptions for a session (schema + semantic validated)."""

    name: str = "session_llm_plan_advice"
    version: str = "0.1.0"

    llm_call_op: Any = None
    parse_op: Any = None

    def __post_init__(self) -> None:
        if self.llm_call_op is None:
            self.llm_call_op = LLMCallOperator()
        if self.parse_op is None:
            self.parse_op = ParseJSONWithSchemaOperator()

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        session_run_dir = inputs.get("session_run_dir") or (Path(ctx.run_dir()) if hasattr(ctx, "run_dir") else None)
        if session_run_dir is None:
            raise ValidationError("SessionLLMPlanAdviceOperator requires session_run_dir or ctx with run_dir()")

        store = SessionStore(Path(session_run_dir))
        state = store.load_session_state()

        # plan revision
        next_rev = (state.latest_plan_rev if state.latest_plan_rev is not None else -1) + 1
        plan_id = f"plan_rev{next_rev:04d}"
        plan_rel = f"session/llm/plan_advice/{plan_id}"
        plan_dir = store.run_dir / plan_rel
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Tool catalog (from registry metadata; execution uses a separately configured registry).
        registry = build_session_plan_registry()
        tool_catalog = build_tool_catalog(registry)
        store.recorder.write_json(f"{plan_rel}/tool_catalog.json", tool_catalog)

        # Build planning context.
        context = build_session_llm_planning_context(
            store,
            tool_catalog=tool_catalog,
            max_topk=int(inputs.get("max_topk", 5)),
            max_pareto=int(inputs.get("max_pareto", 5)),
        )
        store.recorder.write_json(f"{plan_rel}/context.json", context)

        cfg_payload = inputs.get("llm_config") or {}
        if isinstance(cfg_payload, LLMConfig):
            llm_cfg = cfg_payload
        elif isinstance(cfg_payload, Mapping):
            llm_cfg = LLMConfig(
                provider=str(cfg_payload.get("provider", "openai")),
                model=str(cfg_payload.get("model", "gpt-4.1")),
                temperature=float(cfg_payload.get("temperature", 0.2)),
                max_tokens=cfg_payload.get("max_tokens"),
                seed=cfg_payload.get("seed"),
            )
        else:
            raise ValidationError("llm_config must be a mapping or LLMConfig")

        max_repairs = int(inputs.get("max_repairs", 1))

        sys_p, user_p = build_plan_options_prompt(context)
        (plan_dir / "prompt.txt").write_text(user_p, encoding="utf-8")

        request = LLMRequest(system=sys_p, user=user_p, config=llm_cfg, response_schema_name="plan_options")
        stage = f"{plan_rel}/plan"
        call_inputs: dict[str, Any] = {"request": request, "stage": stage}
        if llm_cfg.provider == "mock":
            for key in ("mock_response", "mock_responses", "mock_fn"):
                if key in inputs:
                    call_inputs[key] = inputs[key]
        call_res = self.llm_call_op.run(call_inputs, ctx=ctx)
        raw_text = call_res.outputs["response_text"]
        (plan_dir / "response.txt").write_text(str(raw_text), encoding="utf-8")

        parse_out = self.parse_op.run(
            {"text": raw_text, "schema_name": "plan_options", "validate_fn": validate_llm_plan_options},
            ctx=None,
        ).outputs
        parsed = parse_out["parsed"]
        validation = parse_out["validation"]

        # Repair loop for JSON/schema failures.
        repair_attempt = 0
        while parsed is None and repair_attempt < max_repairs:
            repair_attempt += 1
            prompt = build_repair_prompt(schema=PLAN_OPTIONS_SCHEMA, error=str(validation.get("error")), bad_output=str(raw_text))
            repair_req = LLMRequest(
                system="You are a JSON repair tool.",
                user=prompt,
                config=llm_cfg,
                response_schema_name="plan_options_repair",
            )
            repair_stage = f"{plan_rel}/repair{repair_attempt}"
            repair_inputs: dict[str, Any] = {"request": repair_req, "stage": repair_stage}
            if llm_cfg.provider == "mock":
                mock_repairs = inputs.get("mock_repairs")
                if mock_repairs is not None:
                    repair_inputs["mock_responses"] = mock_repairs
            repair_res = self.llm_call_op.run(repair_inputs, ctx=ctx)
            raw_text = repair_res.outputs["response_text"]
            parse_out = self.parse_op.run(
                {"text": raw_text, "schema_name": "plan_options", "validate_fn": validate_llm_plan_options},
                ctx=None,
            ).outputs
            parsed = parse_out["parsed"]
            validation = parse_out["validation"]

        if parsed is None:
            raise ValidationError(f"failed to parse plan_options: {validation.get('error')}")

        # Semantic validation (tool allowlist + guardrails).
        semantic = validate_plan_options_semantics(parsed, registry=registry)
        if not semantic.ok and repair_attempt < max_repairs:
            # One more repair pass driven by semantic errors (best-effort).
            repair_attempt += 1
            prompt = build_repair_prompt(schema=PLAN_OPTIONS_SCHEMA, error=json.dumps(semantic.to_dict(), indent=2, sort_keys=True), bad_output=json.dumps(parsed, indent=2, sort_keys=True))
            repair_req = LLMRequest(
                system="You are a JSON repair tool.",
                user=prompt,
                config=llm_cfg,
                response_schema_name="plan_options_semantic_repair",
            )
            repair_stage = f"{plan_rel}/semantic_repair{repair_attempt}"
            repair_inputs = {"request": repair_req, "stage": repair_stage}
            if llm_cfg.provider == "mock":
                mock_sem = inputs.get("mock_semantic_repairs")
                if mock_sem is not None:
                    repair_inputs["mock_responses"] = mock_sem
            repair_res = self.llm_call_op.run(repair_inputs, ctx=ctx)
            raw_text = repair_res.outputs["response_text"]
            (plan_dir / "response_semantic_repair.txt").write_text(str(raw_text), encoding="utf-8")
            parse_out = self.parse_op.run(
                {"text": raw_text, "schema_name": "plan_options", "validate_fn": validate_llm_plan_options},
                ctx=None,
            ).outputs
            parsed = parse_out["parsed"]
            validation = parse_out["validation"]
            if parsed is None:
                raise ValidationError(f"failed to parse plan_options semantic repair: {validation.get('error')}")
            semantic = validate_plan_options_semantics(parsed, registry=registry)

        raise_on_invalid_plan(semantic)

        store.recorder.write_json(f"{plan_rel}/plan_options.json", parsed)
        store.recorder.write_json(f"{plan_rel}/validation_report.json", {"parse": validation, "semantic": semantic.to_dict()})
        store.recorder.write_json(
            f"{plan_rel}/status.json",
            {"decision": "pending", "created_at": _utc_now_iso(), "plan_rev": next_rev},
        )

        store.update_session_state(
            lambda s: replace(
                s,
                latest_plan_rev=next_rev,
                plan_history=[
                    *list(s.plan_history),
                    {
                        "rev": next_rev,
                        "timestamp": _utc_now_iso(),
                        "provider": llm_cfg.provider,
                        "model": llm_cfg.model,
                        "decision": "pending",
                    },
                ],
                artifacts_index={
                    **dict(s.artifacts_index),
                    "latest_plan_dir": plan_rel,
                },
            )
        )

        # Refresh meta report so the plan shows up immediately.
        store.recorder.write_text("session/meta_report.md", build_meta_report(store))

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["session_run_dir"] = ArtifactFingerprint(sha256=stable_hash_str(str(store.run_dir)))
        prov.inputs["llm_config"] = ArtifactFingerprint(sha256=stable_hash_json(asdict(llm_cfg)))
        prov.outputs["tool_catalog"] = ArtifactFingerprint(sha256=stable_hash_json(tool_catalog))
        prov.outputs["plan_options"] = ArtifactFingerprint(sha256=stable_hash_json(parsed))
        prov.finish()

        return OperatorResult(outputs={"plan_rev": next_rev, "plan_dir": plan_rel, "plan_options": parsed}, provenance=prov)

