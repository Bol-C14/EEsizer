"""Tool-chain parser/executor inspired by the notebook workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, MutableMapping, Sequence

from .context import ExecutionContext
from .messaging import Message, ToolCall, ToolResult


ToolHandler = Callable[[ToolCall, ExecutionContext, MutableMapping[str, Any]], ToolResult]


class ToolRegistry:
    """Registry of available tool handlers."""

    def __init__(self) -> None:
        self._handlers: MutableMapping[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> ToolHandler:
        if name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._handlers[name]


@dataclass(slots=True)
class ToolChainResult:
    """Wrapper around tool results and the textual sim_output summary."""

    results: Sequence[ToolResult]
    summary: str
    state: MutableMapping[str, Any]


class ToolChainExecutor:
    """Executes tool calls sequentially while sharing mutable state."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(
        self,
        tool_calls: Sequence[ToolCall],
        context: ExecutionContext,
        *,
        state: MutableMapping[str, Any] | None = None,
    ) -> ToolChainResult:
        shared_state: MutableMapping[str, Any] = state if state is not None else {}
        results = []
        summary_chunks: list[str] = []
        for call in tool_calls:
            handler = self.registry.get(call.name)
            result = handler(call, context, shared_state)
            results.append(result)
            summary_chunks.append(_summarize_tool_result(call, result))
        summary = " | ".join(chunk for chunk in summary_chunks if chunk)
        return ToolChainResult(results=tuple(results), summary=summary, state=shared_state)


class ToolChainParser:
    """Parses tool call definitions from agent messages."""

    JSON_BLOCK = re.compile(r"```(?:json)?\n(?P<body>.+?)```", re.DOTALL | re.IGNORECASE)

    def parse(self, messages: Iterable[Message]) -> Sequence[ToolCall]:
        calls: list[ToolCall] = []
        for message in messages:
            if message.tool_calls:
                calls.extend(message.tool_calls)
                continue
            for payload in self._extract_json_blocks(message.content):
                if isinstance(payload, dict) and "name" in payload:
                    calls.append(
                        ToolCall(
                            name=str(payload["name"]),
                            arguments=dict(payload.get("arguments", {})),
                            call_id=str(payload.get("id", f"call_{len(calls)+1}")),
                            description=payload.get("description"),
                        )
                    )
                elif isinstance(payload, list):
                    for idx, item in enumerate(payload, start=1):
                        if isinstance(item, dict) and "name" in item:
                            calls.append(
                                ToolCall(
                                    name=str(item["name"]),
                                    arguments=dict(item.get("arguments", {})),
                                    call_id=str(item.get("id", f"call_{len(calls)+idx}")),
                                    description=item.get("description"),
                                )
                            )
        return calls

    def _extract_json_blocks(self, text: str) -> Sequence[Any]:
        payloads: list[Any] = []
        for match in self.JSON_BLOCK.finditer(text):
            body = match.group("body")
            try:
                payloads.append(json.loads(body))
            except json.JSONDecodeError:
                continue
        return payloads


def _summarize_tool_result(call: ToolCall, result: ToolResult) -> str:
    content = result.content
    if isinstance(content, dict):
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(content.items()))
    else:
        pairs = str(content)
    return f"{call.name}: {pairs}"


__all__ = [
    "ToolChainExecutor",
    "ToolChainParser",
    "ToolChainResult",
    "ToolRegistry",
    "ToolHandler",
]
