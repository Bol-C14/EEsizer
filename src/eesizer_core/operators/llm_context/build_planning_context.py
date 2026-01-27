from __future__ import annotations

from typing import Any, Mapping

from .build_context import build_session_llm_context
from ...runtime.session_store import SessionStore


def build_session_llm_planning_context(
    store: SessionStore,
    *,
    tool_catalog: Mapping[str, Any],
    max_topk: int = 5,
    max_pareto: int = 5,
    max_report_section_lines: int = 60,
) -> dict[str, Any]:
    """Build a compact, deterministic planning context for Step8.

    This extends the Step7 context with the current ToolCatalog.
    """
    base = build_session_llm_context(
        store,
        max_topk=max_topk,
        max_pareto=max_pareto,
        max_report_section_lines=max_report_section_lines,
    )
    return {**dict(base), "tool_catalog": dict(tool_catalog)}

