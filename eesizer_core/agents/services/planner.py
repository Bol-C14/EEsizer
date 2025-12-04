"""Planning service that builds task decomposition prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ...context import ExecutionContext
from ...messaging import Message, MessageRole
from ...netlist import NetlistSummary, summarize_netlist
from ...prompts import PromptLibrary
from ..scoring import OptimizationTargets
from ...providers import LLMResponse


class PlannerService:
    """Encapsulates planning logic for sizing agents."""

    def __init__(
        self,
        prompts: PromptLibrary,
        chat: Callable[..., LLMResponse],
    ) -> None:
        self.prompts = prompts
        self.chat = chat

    def build_plan(
        self,
        context: ExecutionContext,
        netlist_path: Path,
        netlist_text: str,
        goal: str,
        targets: OptimizationTargets,
    ) -> tuple[Sequence[Message], NetlistSummary]:
        """Return planning messages plus the computed netlist summary."""

        summary = summarize_netlist(netlist_path, netlist_text)
        template = self.prompts.load("task_decomposition")
        plan_note = template.render(goal=goal, netlist_summary=summary.describe())
        target_template = self.prompts.load("target_extraction")
        target_note = target_template.render(
            goal=goal,
            target_gain_db=targets.gain_db,
            target_power_mw=targets.power_mw,
        )
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=plan_note,
            name="planner",
        )
        user_message = Message(
            role=MessageRole.USER,
            content=target_note,
            name="target_extraction",
        )
        response = self.chat([system_message, user_message], response_name="plan")
        messages = (system_message, user_message, response.message)
        return messages, summary


__all__ = ["PlannerService"]
