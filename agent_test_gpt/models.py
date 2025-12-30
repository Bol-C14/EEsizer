"""Typed data structures for parsed LLM outputs and run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskQuestions:
    """Structured task prompts returned by the LLM."""

    type_question: str
    node_question: str
    sim_question: str
    sizing_question: str

    @classmethod
    def from_json(cls, payload: Dict) -> "TaskQuestions":
        if not isinstance(payload, dict):
            raise ValueError("tasks payload must be a dict")
        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError("tasks.questions must be a list")

        merged: Dict[str, str] = {}
        for item in questions:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if k in {"type_question", "node_question", "sim_question", "sizing_question"} and isinstance(v, str):
                    merged[k] = v.strip()

        missing = [k for k in ("type_question", "node_question", "sim_question", "sizing_question") if k not in merged]
        if missing:
            raise ValueError(f"tasks missing required keys: {missing}")

        return cls(
            type_question=merged["type_question"],
            node_question=merged["node_question"],
            sim_question=merged["sim_question"],
            sizing_question=merged["sizing_question"],
        )


@dataclass
class TargetValues:
    """Normalized numeric targets (float or None) and pass flags."""

    targets: Dict[str, Optional[float]] = field(default_factory=dict)
    passes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict]:
        return {"targets": self.targets, "passes": self.passes}
