from __future__ import annotations

from typing import Any, Mapping
import json

from ...contracts.llm_insights import INSIGHTS_SCHEMA


SYSTEM = """You are ReportInterpreterAgent.

You read an engineering optimization session context and produce a structured, audit-friendly analysis.

Hard rules:
- Output MUST be strict JSON (no markdown, no commentary).
- Output MUST match the provided JSON schema exactly (required keys present; correct types).
- Do NOT propose edits to netlists, include paths, files, or any tool execution.
- Every claim MUST be backed by at least one evidence entry that is a path from context.evidence.
"""


def build_report_interpret_prompt(context: Mapping[str, Any]) -> tuple[str, str]:
    schema_text = json.dumps(INSIGHTS_SCHEMA, indent=2, sort_keys=True)
    ctx_text = json.dumps(dict(context), indent=2, sort_keys=True)

    user = "\n".join(
        [
            "Return a JSON object that matches this schema:",
            schema_text,
            "",
            "Context (JSON):",
            ctx_text,
            "",
            "Notes:",
            "- summary should be 200-400 words max.",
            "- tradeoffs should focus on (power_w vs ugbw_hz) and PM constraints when available.",
            "- sensitivity_rank should cite the sensitivity.json evidence when available.",
            "- robustness_notes should cite robust_topk/robust_pareto evidence when available.",
        ]
    )
    return SYSTEM, user

