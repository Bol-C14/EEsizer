"""Core messaging primitives shared by all agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


class MessageRole(str, Enum):
    """Canonical set of roles aligned with OpenAI/Gemini/Claude chat formats."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """Represents a function/tool invocation emitted by an agent."""

    name: str
    arguments: Mapping[str, Any]
    call_id: Optional[str] = None
    description: str | None = None


@dataclass(slots=True)
class ToolResult:
    """Holds structured output returned to the agent after running a tool."""

    call_id: str
    content: Any
    ok: bool = True
    diagnostics: MutableMapping[str, Any] | None = None


@dataclass(slots=True)
class Message:
    """Envelope for every prompt/response exchanged across agents and tools."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    attachments: Sequence[str] = field(default_factory=tuple)
    tags: Mapping[str, str] | None = None

    def with_tool_call(self, tool_call: ToolCall) -> "Message":
        """Return a new message containing the provided tool call."""

        calls = list(self.tool_calls)
        calls.append(tool_call)
        return Message(
            role=self.role,
            content=self.content,
            name=self.name,
            tool_calls=tuple(calls),
            attachments=self.attachments,
            tags=self.tags,
        )


@dataclass(slots=True)
class MessageBundle:
    """Convenience wrapper around ordered chat history slices."""

    messages: List[Message]

    def as_dict(self) -> List[Dict[str, Any]]:
        """Return a list that mirrors OpenAI-compatible payloads."""

        result: List[Dict[str, Any]] = []
        for message in self.messages:
            payload: Dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.name:
                payload["name"] = message.name
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "function": {"name": call.name, "arguments": call.arguments},
                        **({"description": call.description} if call.description else {}),
                    }
                    for call in message.tool_calls
                ]
            result.append(payload)
        return result

    @classmethod
    def from_iterable(cls, iterable: Iterable[Message]) -> "MessageBundle":
        """Build a bundle from any iterable of messages."""

        return cls(messages=list(iterable))
