"""LLM provider abstractions with live and recorded backends."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .config import AgentConfig, ToolConfig
from .messaging import Message, MessageBundle, MessageRole, ToolCall


@dataclass(slots=True)
class LLMResponse:
    """Normalized chat response returned by provider backends."""

    message: Message
    raw: Mapping[str, object] | None = None


class LLMProvider(Protocol):
    """Minimal interface shared by live and recorded providers."""

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> LLMResponse:
        ...


class RecordedProvider:
    """Fixture-backed provider that mirrors notebook prompts without live calls."""

    def __init__(
        self,
        model_key: str,
        *,
        recordings: Mapping[str, Mapping[str, str]] | None = None,
        default_tool_calls: Sequence[ToolCall] | None = None,
    ) -> None:
        self.model_key = model_key
        self.recordings = recordings or {}
        self.default_tool_calls = tuple(default_tool_calls or ())

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,  # noqa: ARG002 - parity with protocol
    ) -> LLMResponse:
        model_blob = self.recordings.get(self.model_key, {})
        text = model_blob.get(response_name or "", "") if model_blob else ""
        if not text:
            text = model_blob.get("default", "") if model_blob else ""
        if not text and messages:
            text = messages[-1].content
        tool_calls: tuple[ToolCall, ...] = ()
        if response_name == "tool_plan":
            tool_calls = self.default_tool_calls
        message = Message(role=MessageRole.ASSISTANT, content=text, tool_calls=tool_calls)
        return LLMResponse(message=message, raw={"recorded": True})


class OpenAIProvider:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str, *, api_key: str, base_url: str | None = None, **kwargs):
        spec = importlib.util.find_spec("openai")
        if spec is None:
            raise RuntimeError("openai package is required for live OpenAI calls")
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,  # noqa: ARG002 - live calls ignore hint
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": MessageBundle.from_iterable(messages).as_dict(),
        }
        if tools:
            payload["tools"] = list(tools)
        completion = self.client.chat.completions.create(**payload)
        choice = completion.choices[0].message
        tool_calls: list[ToolCall] = []
        for call in choice.tool_calls or []:
            arguments = call.function.arguments if hasattr(call.function, "arguments") else {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            tool_calls.append(
                ToolCall(
                    name=str(call.function.name),
                    arguments=arguments,
                    call_id=call.id,
                    description=getattr(call, "description", None),
                )
            )
        message = Message(
            role=MessageRole.ASSISTANT,
            content=choice.content or "",
            tool_calls=tuple(tool_calls),
        )
        return LLMResponse(message=message, raw=completion.model_dump())


class AnthropicProvider:
    """Wrapper for Anthropic Messages API with tool support."""

    def __init__(self, model: str, *, api_key: str, base_url: str | None = None):
        spec = importlib.util.find_spec("anthropic")
        if spec is None:
            raise RuntimeError("anthropic package is required for live Claude calls")
        import anthropic

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,  # noqa: ARG002 - live calls ignore hint
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> LLMResponse:
        content_messages = [
            {"role": message.role.value, "content": message.content} for message in messages
        ]
        completion = self.client.messages.create(
            model=self.model,
            messages=content_messages,
            tools=list(tools or ()),
        )
        text_chunks = [
            block.text
            for block in completion.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        tool_calls: list[ToolCall] = []
        for block in completion.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=str(getattr(block, "name", "")),
                        arguments=getattr(block, "input", {}),
                        call_id=str(getattr(block, "id", "")),
                    )
                )
        message = Message(
            role=MessageRole.ASSISTANT,
            content="\n".join(text_chunks),
            tool_calls=tuple(tool_calls),
        )
        return LLMResponse(message=message, raw=getattr(completion, "model_dump", lambda: {})())


class GeminiProvider:
    """Wrapper for Gemini chat responses using google-generativeai."""

    def __init__(self, model: str, *, api_key: str):
        spec = importlib.util.find_spec("google.generativeai")
        if spec is None:
            raise RuntimeError("google-generativeai package is required for live Gemini calls")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,  # noqa: ARG002 - live calls ignore hint
        tools: Sequence[Mapping[str, object]] | None = None,  # noqa: ARG002 - Gemini tool-calls not wired yet
    ) -> LLMResponse:
        history = [f"{message.role.value}: {message.content}" for message in messages[:-1]]
        prompt = messages[-1].content if messages else ""
        chat_session = self.model.start_chat(history=history)
        response = chat_session.send_message(prompt)
        text = (
            response.text() if callable(getattr(response, "text", None)) else str(getattr(response, "text", ""))
        )
        message = Message(role=MessageRole.ASSISTANT, content=text)
        raw = getattr(response, "_result", None)
        return LLMResponse(message=message, raw=raw)


def build_provider(
    agent_config: AgentConfig,
    tools: Mapping[str, ToolConfig] | None,
    *,
    recordings: Mapping[str, Mapping[str, str]] | None = None,
    default_tool_calls: Sequence[ToolCall] | None = None,
    force_live: bool | None = None,
) -> LLMProvider:
    """Select a provider based on tool kind and environment flags.

    Live calls are enabled when any of the following are true:
    - ``force_live`` is explicitly provided
    - ``EESIZER_LIVE_LLM`` is set to a truthy value
    - an API key is present for at least one referenced tool
    """

    def _env_truthy() -> bool:
        raw = os.getenv("EESIZER_LIVE_LLM")
        if raw is None:
            return False
        return raw.strip().lower() not in {"", "0", "false", "no"}

    def _has_credentials(tool_map: Mapping[str, ToolConfig]) -> bool:
        for tool_name in agent_config.tools:
            cfg = tool_map.get(tool_name)
            if cfg and cfg.credentials.get("api_key"):
                return True
        return False

    tool_map = tools or {}
    live_flag = force_live if force_live is not None else _env_truthy() or _has_credentials(tool_map)
    if not live_flag:
        return RecordedProvider(
            agent_config.name,
            recordings=recordings,
            default_tool_calls=default_tool_calls,
        )

    for tool_name in agent_config.tools:
        cfg = tool_map.get(tool_name)
        if not cfg:
            continue
        if cfg.kind in {"openai", "gpt"}:
            api_key = str(cfg.credentials.get("api_key", ""))
            base_url = cfg.parameters.get("base_url")
            return OpenAIProvider(agent_config.model, api_key=api_key, base_url=base_url)
        if cfg.kind in {"anthropic", "claude"}:
            api_key = str(cfg.credentials.get("api_key", ""))
            base_url = cfg.parameters.get("base_url")
            return AnthropicProvider(agent_config.model, api_key=api_key, base_url=base_url)
        if cfg.kind in {"gemini", "google"}:
            api_key = str(cfg.credentials.get("api_key", ""))
            return GeminiProvider(agent_config.model, api_key=api_key)

    return RecordedProvider(
        agent_config.name,
        recordings=recordings,
        default_tool_calls=default_tool_calls,
    )


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "RecordedProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "build_provider",
]
