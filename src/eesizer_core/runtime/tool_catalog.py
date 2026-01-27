from __future__ import annotations

from typing import Any

from ..contracts.provenance import stable_hash_json
from .tool_registry import ToolRegistry


def build_tool_catalog(registry: ToolRegistry) -> dict[str, Any]:
    """Export a deterministic, LLM-friendly tool catalog from a ToolRegistry."""
    tools: list[dict[str, Any]] = []
    for name in registry.names():
        spec = registry.spec(name)
        tools.append(
            {
                "name": name,
                "description": spec.description,
                "params_schema": dict(spec.schema or {}),
                "cost_model": dict(spec.cost_model or {}),
                "side_effects": list(spec.side_effects or []),
                "constraints": list(spec.constraints or []),
                "io": {
                    "inputs": list((spec.io or {}).get("inputs") or []),
                    "outputs": list((spec.io or {}).get("outputs") or []),
                },
            }
        )
    catalog: dict[str, Any] = {"tools": tools}
    catalog["sha256"] = stable_hash_json(catalog)
    return catalog

