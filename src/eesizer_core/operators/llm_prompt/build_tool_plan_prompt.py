from __future__ import annotations

from typing import Any, Mapping
import json

from ...contracts.llm_proposal import PROPOSAL_SCHEMA


SYSTEM = """You are ToolPlannerAgent.

You turn a structured analysis into executable, safe configuration/spec deltas.

Hard rules:
- Output MUST be strict JSON (no markdown, no commentary).
- Output MUST match the provided JSON schema exactly.
- You may ONLY propose spec_delta/cfg_delta changes that are safe:
  - spec_delta: objective updates (target/weight/tol/sense) or add/remove objectives
  - cfg_delta: budget fields and grid_search/corner_validate fields only
- You MUST NOT propose netlist edits, include-path changes, file writes, or tool execution.
- Provide 3 options minimum (A/B/C style), each with clear intent, risks, and a rough budget estimate.
"""


def build_tool_plan_prompt(*, insights: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[str, str]:
    schema_text = json.dumps(PROPOSAL_SCHEMA, indent=2, sort_keys=True)
    insights_text = json.dumps(dict(insights), indent=2, sort_keys=True)
    # Provide a small subset of context for planning (current spec/cfg + evidence list).
    context_subset = {
        "session": context.get("session"),
        "spec": context.get("spec"),
        "cfg": context.get("cfg"),
        "evidence": context.get("evidence"),
    }
    ctx_text = json.dumps(context_subset, indent=2, sort_keys=True)

    user = "\n".join(
        [
            "Return a JSON object that matches this schema:",
            schema_text,
            "",
            "Structured insights (JSON):",
            insights_text,
            "",
            "Current session/spec/cfg (JSON):",
            ctx_text,
            "",
            "Delta format guidance:",
            "- spec_delta: {\"objectives\": [{\"metric\": \"phase_margin_deg\", \"op\": \"target\", \"value\": 60.0}], \"notes\": {}}",
            "- cfg_delta: {\"budget\": {\"max_iterations\": 25}, \"notes\": {\"grid_search\": {\"levels\": 10, \"span_mul\": 8.0}}}",
        ]
    )
    return SYSTEM, user

