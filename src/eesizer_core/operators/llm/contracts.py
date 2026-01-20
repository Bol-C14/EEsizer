from __future__ import annotations

"""Lightweight LLM request/response structures for operator use."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LLMConfig:
    """Minimal configuration for a chat-style LLM call."""

    provider: str = "openai"
    model: str = "gpt-4.1"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class LLMRequest:
    """Structured LLM request payload."""

    system: str
    user: str
    config: LLMConfig
    response_schema_name: Optional[str] = None


@dataclass(frozen=True)
class LLMResponse:
    """Structured LLM response payload."""

    text: str
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
