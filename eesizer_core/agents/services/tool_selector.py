"""Tool selection service that builds simulator/tool plans."""

from __future__ import annotations

import json
from typing import Callable, Mapping, Sequence

from ...context import ExecutionContext
from ...messaging import Message, MessageRole
from ...prompts import PromptLibrary
from ...providers import LLMResponse


class ToolSelectionService:
    """Encapsulates tool planning prompts and parsing."""

    def __init__(
        self,
        prompts: PromptLibrary,
        chat: Callable[..., LLMResponse],
    ) -> None:
        self.prompts = prompts
        self.chat = chat

    def select_tools(
        self,
        context: ExecutionContext,
        *,
        agent_name: str,
        netlist_summary: str,
        default_tool_plan: Sequence[Mapping[str, object]],
        tool_schemas: Sequence[Mapping[str, object]],
    ) -> Sequence[Message]:
        """Return messages describing the simulator/tool choices."""

        payload = json.dumps(default_tool_plan, indent=2)
        template = self.prompts.load("simulation_planning")
        content = template.render(
            agent_name=agent_name,
            netlist_summary=netlist_summary,
            tool_blueprint=payload,
        )
        request = Message(role=MessageRole.USER, content=content, name="tool_selector")
        response = self.chat(
            [request],
            response_name="tool_plan",
            tools=tool_schemas,
        )
        messages = (request, response.message)
        return messages


__all__ = ["ToolSelectionService"]
