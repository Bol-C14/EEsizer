from __future__ import annotations

from typing import Any, Mapping
import json

from ...contracts.plan_options import PLAN_OPTIONS_SCHEMA


SYSTEM = """You are ToolPlannerAgent.

You produce multi-step Plan options composed of whitelisted tool calls.

Hard rules:
- Output MUST be strict JSON (no markdown, no commentary).
- Output MUST match the provided JSON schema exactly.
- You may ONLY use tools listed in tool_catalog.tools[].name.
- For each action, inputs/outputs MUST exactly match tool_catalog.tools[].io when provided.
- Do NOT propose any netlist edits, include-path changes, file writes, or shell commands.
- Keep plans short (<= 6 actions) and deterministic.
"""


def build_plan_options_prompt(context: Mapping[str, Any]) -> tuple[str, str]:
    schema_text = json.dumps(PLAN_OPTIONS_SCHEMA, indent=2, sort_keys=True)
    ctx_text = json.dumps(dict(context), indent=2, sort_keys=True)
    user = "\n".join(
        [
            "Return a JSON object that matches this schema:",
            schema_text,
            "",
            "Planning context (JSON):",
            ctx_text,
            "",
            "Output guidance:",
            "- Provide at least 3 options (A/B/C style).",
            "- Prefer: update_cfg/update_spec -> run_grid_search -> run_corner_validate.",
            "- Mark robustness-sensitive steps with requires_approval=true.",
        ]
    )
    return SYSTEM, user

