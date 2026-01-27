from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ValidationError


INSIGHTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "tradeoffs", "sensitivity_rank", "robustness_notes", "recommended_actions"],
    "properties": {
        "summary": {"type": "string"},
        "tradeoffs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["x", "y", "pattern", "evidence"],
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "pattern": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "sensitivity_rank": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["param_id", "impact", "evidence"],
                "properties": {
                    "param_id": {"type": "string"},
                    "impact": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "robustness_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["worst_corner_id", "why", "evidence"],
                "properties": {
                    "worst_corner_id": {"type": "string"},
                    "why": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "recommended_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action_type", "rationale"],
                "properties": {
                    "action_type": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def validate_llm_insights(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError("insights must be a JSON object")

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValidationError("insights.summary must be a non-empty string")

    def _validate_items(key: str, required: tuple[str, ...]) -> list[dict[str, Any]]:
        items = payload.get(key)
        if not isinstance(items, list):
            raise ValidationError(f"insights.{key} must be a list")
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for req in required:
                if req not in item:
                    raise ValidationError(f"insights.{key} item missing '{req}'")
            out.append(dict(item))
        return out

    tradeoffs = _validate_items("tradeoffs", ("x", "y", "pattern", "evidence"))
    for item in tradeoffs:
        item["evidence"] = _as_str_list(item.get("evidence"))

    sensitivity = _validate_items("sensitivity_rank", ("param_id", "impact", "evidence"))
    for item in sensitivity:
        item["evidence"] = _as_str_list(item.get("evidence"))

    robustness = _validate_items("robustness_notes", ("worst_corner_id", "why", "evidence"))
    for item in robustness:
        item["evidence"] = _as_str_list(item.get("evidence"))

    actions = _validate_items("recommended_actions", ("action_type", "rationale"))

    return {
        "summary": summary.strip(),
        "tradeoffs": tradeoffs,
        "sensitivity_rank": sensitivity,
        "robustness_notes": robustness,
        "recommended_actions": actions,
    }

