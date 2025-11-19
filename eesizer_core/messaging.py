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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolCall.name must be provided")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("ToolCall.arguments must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.call_id:
            payload["id"] = self.call_id
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolCall":
        return cls(
            name=str(payload.get("name", "")),
            arguments=dict(payload.get("arguments", {})),
            call_id=payload.get("id"),
            description=payload.get("description"),
        )


@dataclass(slots=True)
class ToolResult:
    """Holds structured output returned to the agent after running a tool."""

    call_id: str
    content: Any
    ok: bool = True
    diagnostics: MutableMapping[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "call_id": self.call_id,
            "content": self.content,
            "ok": self.ok,
        }
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload


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

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.attachments:
            payload["attachments"] = list(self.attachments)
        if self.tags:
            payload["tags"] = dict(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Message":
        role_value = payload.get("role", MessageRole.USER.value)
        role = role_value if isinstance(role_value, MessageRole) else MessageRole(role_value)
        tool_calls_raw = payload.get("tool_calls", [])

        def _to_tool_call(raw: Mapping[str, Any]) -> ToolCall:
            if "function" in raw:
                function = raw.get("function", {})
                return ToolCall(
                    name=str(function.get("name", "")),
                    arguments=dict(function.get("arguments", {})),
                    call_id=raw.get("id"),
                    description=raw.get("description"),
                )
            return ToolCall.from_dict(raw)

        tool_calls = tuple(_to_tool_call(raw) for raw in tool_calls_raw)
        attachments = tuple(payload.get("attachments", ()))
        tags_field = payload.get("tags")
        tags = dict(tags_field) if isinstance(tags_field, Mapping) else None
        return cls(
            role=role,
            content=str(payload.get("content", "")),
            name=payload.get("name"),
            tool_calls=tool_calls,
            attachments=attachments,
            tags=tags,
        )


@dataclass(slots=True)
class MessageBundle:
    """Convenience wrapper around ordered chat history slices."""

    messages: List[Message]

    def as_dict(self) -> List[Dict[str, Any]]:
        """Return a list that mirrors OpenAI-compatible payloads."""

        result: List[Dict[str, Any]] = []
        for message in self.messages:
            payload = message.to_dict()
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

    @classmethod
    def from_dicts(cls, payload: Iterable[Mapping[str, Any]]) -> "MessageBundle":
        """Build a bundle from dictionaries that resemble API payloads."""

        return cls(messages=[Message.from_dict(item) for item in payload])

    def validate_tool_schema(self) -> None:
        """Ensure every tool call includes the minimum schema required by providers."""

        for message in self.messages:
            for call in message.tool_calls:
                if not call.name:
                    raise ValueError("Tool call names cannot be empty")
                if not isinstance(call.arguments, Mapping):
                    raise TypeError("Tool call arguments must be JSON-serializable mappings")
