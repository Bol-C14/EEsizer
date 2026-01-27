from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json

from ..analysis.session_report import build_meta_report
from ..contracts.errors import ValidationError
from ..contracts.llm_insights import INSIGHTS_SCHEMA, validate_llm_insights
from ..contracts.llm_proposal import PROPOSAL_SCHEMA, validate_llm_proposal
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ..operators.llm.llm_call import LLMCallOperator
from ..operators.llm.contracts import LLMConfig, LLMRequest
from ..operators.llm_context import build_session_llm_context
from ..operators.llm_parse import ParseJSONWithSchemaOperator, build_repair_prompt
from ..operators.llm_prompt import build_report_interpret_prompt, build_tool_plan_prompt
from ..runtime.session_store import SessionStore


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _cache_key(payload: Any) -> str:
    # Bare sha256 hex; stable_hash_json already normalizes containers.
    return stable_hash_json(payload)


def _load_cache_index(store: SessionStore) -> dict[str, Any]:
    path = store.session_dir / "llm" / "cache_index.json"
    payload = _read_json(path)
    if isinstance(payload, dict):
        return payload
    return {}


def _write_cache_index(store: SessionStore, index: Mapping[str, Any]) -> None:
    path = store.session_dir / "llm" / "cache_index.json"
    _write_json(path, dict(index))


def _render_narrative(insights: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# LLM Interpretation (Optional)")
    lines.append("")
    summary = insights.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary.strip())
        lines.append("")
    actions = insights.get("recommended_actions") if isinstance(insights.get("recommended_actions"), list) else []
    if actions:
        lines.append("## Recommended Actions (LLM)")
        for item in actions:
            if not isinstance(item, Mapping):
                continue
            a = item.get("action_type")
            r = item.get("rationale")
            if a and r:
                lines.append(f"- {a}: {r}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionLLMAdviseOperator(Operator):
    """Generate insights.json + proposal.json for a session (with caching + schema validation)."""

    name: str = "session_llm_advise"
    version: str = "0.1.0"

    llm_call_op: Any = None
    parse_op: Any = None

    def __post_init__(self) -> None:
        if self.llm_call_op is None:
            self.llm_call_op = LLMCallOperator()
        if self.parse_op is None:
            self.parse_op = ParseJSONWithSchemaOperator()

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        # ctx is expected to be a RunContext; stage dirs are created under ctx.run_dir().
        session_run_dir = inputs.get("session_run_dir") or (Path(ctx.run_dir()) if hasattr(ctx, "run_dir") else None)
        if session_run_dir is None:
            raise ValidationError("SessionLLMAdviseOperator requires session_run_dir or ctx with run_dir()")

        store = SessionStore(Path(session_run_dir))
        state = store.load_session_state()

        # Determine next advice revision.
        next_rev = (state.latest_advice_rev if state.latest_advice_rev is not None else -1) + 1
        advice_id = f"advice_rev{next_rev:04d}"
        advice_rel = f"session/llm/advice/{advice_id}"
        advice_dir = store.run_dir / advice_rel
        advice_dir.mkdir(parents=True, exist_ok=True)

        # Build deterministic context from artifacts.
        context = build_session_llm_context(
            store,
            max_topk=int(inputs.get("max_topk", 5)),
            max_pareto=int(inputs.get("max_pareto", 5)),
        )
        store.recorder.write_json(f"session/llm/context_rev{next_rev:04d}.json", context)
        store.recorder.write_json(f"{advice_rel}/context.json", context)

        # LLM config
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
        cache_index = _load_cache_index(store)

        # --- Agent A: interpret ---
        sys_a, user_a = build_report_interpret_prompt(context)
        (advice_dir / "prompt_interpret.txt").write_text(user_a, encoding="utf-8")

        interpret_key = _cache_key(
            {
                "kind": "interpret",
                "context_sha256": stable_hash_json(context),
                "system": sys_a,
                "user": user_a,
                "config": asdict(llm_cfg),
            }
        )
        interpret_cached = cache_index.get(interpret_key)
        insights: dict[str, Any] | None = None
        interpret_validation: dict[str, Any] = {"ok": False, "schema": "llm_insights", "error": "not_run"}

        if isinstance(interpret_cached, str):
            cached_path = store.run_dir / interpret_cached
            cached_payload = _read_json(cached_path)
            if cached_payload is not None:
                try:
                    insights = validate_llm_insights(cached_payload)
                    interpret_validation = {
                        "ok": True,
                        "schema": "llm_insights",
                        "error": None,
                        "cache": True,
                        "cache_path": interpret_cached,
                    }
                    (advice_dir / "response_interpret.txt").write_text(
                        f"[cache_hit] reused {interpret_cached}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    insights = None

        if insights is None:
            request = LLMRequest(system=sys_a, user=user_a, config=llm_cfg, response_schema_name="llm_insights")
            stage = f"{advice_rel}/interpret"
            call_inputs = {"request": request, "stage": stage}
            if llm_cfg.provider == "mock":
                # allow passing mock responses for tests
                for key in ("mock_response", "mock_responses", "mock_fn"):
                    if key in inputs:
                        call_inputs[key] = inputs[key]
            call_res = self.llm_call_op.run(call_inputs, ctx=ctx)
            raw_text = call_res.outputs["response_text"]
            (advice_dir / "response_interpret.txt").write_text(str(raw_text), encoding="utf-8")
            parse_out = self.parse_op.run(
                {"text": raw_text, "schema_name": "llm_insights", "validate_fn": validate_llm_insights},
                ctx=None,
            ).outputs
            insights = parse_out["parsed"]
            interpret_validation = parse_out["validation"]

            repair_attempt = 0
            while insights is None and repair_attempt < max_repairs:
                repair_attempt += 1
                prompt = build_repair_prompt(schema=INSIGHTS_SCHEMA, error=str(interpret_validation.get("error")), bad_output=str(raw_text))
                repair_request = LLMRequest(
                    system="You are a JSON repair tool.",
                    user=prompt,
                    config=llm_cfg,
                    response_schema_name="llm_insights_repair",
                )
                stage = f"{advice_rel}/interpret_repair{repair_attempt}"
                repair_inputs = {"request": repair_request, "stage": stage}
                if llm_cfg.provider == "mock":
                    mock_repairs = inputs.get("mock_repairs")
                    if mock_repairs is not None:
                        repair_inputs["mock_responses"] = mock_repairs
                repair_res = self.llm_call_op.run(repair_inputs, ctx=ctx)
                raw_text = repair_res.outputs["response_text"]
                parse_out = self.parse_op.run(
                    {"text": raw_text, "schema_name": "llm_insights", "validate_fn": validate_llm_insights},
                    ctx=None,
                ).outputs
                insights = parse_out["parsed"]
                interpret_validation = parse_out["validation"]

            if insights is None:
                raise ValidationError(f"failed to parse llm_insights: {interpret_validation.get('error')}")

            store.recorder.write_json(f"{advice_rel}/insights.json", insights)

        # Always materialize in this advice dir and update cache mapping to the latest copy.
        store.recorder.write_json(f"{advice_rel}/insights.json", insights)
        cache_index[interpret_key] = f"{advice_rel}/insights.json"

        # --- Agent B: plan ---
        sys_b, user_b = build_tool_plan_prompt(insights=insights, context=context)
        (advice_dir / "prompt_plan.txt").write_text(user_b, encoding="utf-8")

        plan_key = _cache_key(
            {
                "kind": "plan",
                "context_sha256": stable_hash_json(context),
                "insights_sha256": stable_hash_json(insights),
                "system": sys_b,
                "user": user_b,
                "config": asdict(llm_cfg),
            }
        )
        plan_cached = cache_index.get(plan_key)
        proposal: dict[str, Any] | None = None
        plan_validation: dict[str, Any] = {"ok": False, "schema": "llm_proposal", "error": "not_run"}

        if isinstance(plan_cached, str):
            cached_path = store.run_dir / plan_cached
            cached_payload = _read_json(cached_path)
            if cached_payload is not None:
                try:
                    proposal = validate_llm_proposal(cached_payload)
                    plan_validation = {
                        "ok": True,
                        "schema": "llm_proposal",
                        "error": None,
                        "cache": True,
                        "cache_path": plan_cached,
                    }
                    (advice_dir / "response_plan.txt").write_text(
                        f"[cache_hit] reused {plan_cached}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    proposal = None

        if proposal is None:
            request = LLMRequest(system=sys_b, user=user_b, config=llm_cfg, response_schema_name="llm_proposal")
            stage = f"{advice_rel}/plan"
            call_inputs = {"request": request, "stage": stage}
            if llm_cfg.provider == "mock":
                mock_plan = inputs.get("mock_plan")
                if mock_plan is not None:
                    call_inputs["mock_response"] = mock_plan
            call_res = self.llm_call_op.run(call_inputs, ctx=ctx)
            raw_text = call_res.outputs["response_text"]
            (advice_dir / "response_plan.txt").write_text(str(raw_text), encoding="utf-8")

            parse_out = self.parse_op.run(
                {"text": raw_text, "schema_name": "llm_proposal", "validate_fn": validate_llm_proposal},
                ctx=None,
            ).outputs
            proposal = parse_out["parsed"]
            plan_validation = parse_out["validation"]

            repair_attempt = 0
            while proposal is None and repair_attempt < max_repairs:
                repair_attempt += 1
                prompt = build_repair_prompt(schema=PROPOSAL_SCHEMA, error=str(plan_validation.get("error")), bad_output=str(raw_text))
                repair_request = LLMRequest(
                    system="You are a JSON repair tool.",
                    user=prompt,
                    config=llm_cfg,
                    response_schema_name="llm_proposal_repair",
                )
                stage = f"{advice_rel}/plan_repair{repair_attempt}"
                repair_inputs = {"request": repair_request, "stage": stage}
                if llm_cfg.provider == "mock":
                    mock_plan_repairs = inputs.get("mock_plan_repairs")
                    if mock_plan_repairs is not None:
                        repair_inputs["mock_responses"] = mock_plan_repairs
                repair_res = self.llm_call_op.run(repair_inputs, ctx=ctx)
                raw_text = repair_res.outputs["response_text"]
                parse_out = self.parse_op.run(
                    {"text": raw_text, "schema_name": "llm_proposal", "validate_fn": validate_llm_proposal},
                    ctx=None,
                ).outputs
                proposal = parse_out["parsed"]
                plan_validation = parse_out["validation"]

            if proposal is None:
                raise ValidationError(f"failed to parse llm_proposal: {plan_validation.get('error')}")

            store.recorder.write_json(f"{advice_rel}/proposal.json", proposal)

        store.recorder.write_json(f"{advice_rel}/proposal.json", proposal)
        cache_index[plan_key] = f"{advice_rel}/proposal.json"

        validation_payload = {
            "interpret": interpret_validation,
            "plan": plan_validation,
            "cache_keys": {"interpret": interpret_key, "plan": plan_key},
            "provider": llm_cfg.provider,
            "model": llm_cfg.model,
        }
        store.recorder.write_json(f"{advice_rel}/validation.json", validation_payload)
        store.recorder.write_json(
            f"{advice_rel}/status.json",
            {
                "decision": "pending",
                "created_at": _utc_now_iso(),
                "advice_rev": next_rev,
            },
        )

        # Update cache and session state.
        _write_cache_index(store, cache_index)

        store.update_session_state(
            lambda s: replace(
                s,
                latest_advice_rev=next_rev,
                advice_history=[
                    *list(s.advice_history),
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
                    "latest_advice_dir": advice_rel,
                    "latest_llm_narrative": "session/llm/narrative.md",
                },
            )
        )

        # Write narrative + refresh meta_report so the link is visible.
        narrative = _render_narrative(insights)
        store.recorder.write_text("session/llm/narrative.md", narrative)
        store.recorder.write_text("session/meta_report.md", build_meta_report(store))

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["session_run_dir"] = ArtifactFingerprint(sha256=stable_hash_str(str(store.run_dir)))
        prov.outputs["insights"] = ArtifactFingerprint(sha256=stable_hash_json(insights))
        prov.outputs["proposal"] = ArtifactFingerprint(sha256=stable_hash_json(proposal))
        prov.finish()

        return OperatorResult(
            outputs={
                "advice_rev": next_rev,
                "advice_dir": advice_rel,
                "insights": insights,
                "proposal": proposal,
                "validation": validation_payload,
            },
            provenance=prov,
        )
