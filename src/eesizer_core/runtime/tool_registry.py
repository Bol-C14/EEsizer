"""A small registry to map plan Actions to deterministic execution functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from ..contracts.errors import ValidationError


ToolFn = Callable[[Mapping[str, Any], Any, Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class ToolSpec:
    fn: ToolFn
    schema: Optional[dict] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, op_name: str, fn: ToolFn, schema: Optional[dict] = None) -> None:
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("op_name must be a non-empty string")
        if op_name in self._tools:
            raise ValidationError(f"tool already registered: {op_name}")
        self._tools[op_name] = ToolSpec(fn=fn, schema=schema)

    def has(self, op_name: str) -> bool:
        return op_name in self._tools

    def execute(self, op_name: str, inputs: Mapping[str, Any], ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = self._tools.get(op_name)
        if spec is None:
            raise ValidationError(f"unknown tool op: {op_name}")
        out = spec.fn(inputs, ctx, params)
        if not isinstance(out, Mapping):
            raise ValidationError(f"tool '{op_name}' must return a mapping")
        return out
