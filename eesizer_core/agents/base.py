"""Abstract base class for EEsizer agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence

from ..config import AgentConfig
from ..context import ExecutionContext
from ..messaging import Message


@dataclass(slots=True)
class AgentMetadata:
    """Lightweight descriptor exposed via API and telemetry."""

    name: str
    model_family: str
    description: str
    capabilities: Sequence[str]


@dataclass(slots=True)
class AgentResult:
    """Structured return payload for orchestrators."""

    success: bool
    messages: Sequence[Message]
    artifacts: Mapping[str, str]
    metrics: Mapping[str, float] | None = None


class SupportsAgentIO(Protocol):
    """Protocol for objects that can convert themselves into messages."""

    def to_messages(self) -> Iterable[Message]:
        ...


class Agent(ABC):
    """Lifecycle shared by notebook-derived and production agents."""

    metadata: AgentMetadata

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def build_plan(self, context: ExecutionContext) -> Sequence[Message]:
        """Return the reasoning prompts needed for task decomposition."""

    @abstractmethod
    def select_tools(self, context: ExecutionContext, history: Sequence[Message]) -> Sequence[Message]:
        """Choose simulation steps / measurement utilities based on planning output."""

    @abstractmethod
    def run_simulation(self, context: ExecutionContext) -> MutableMapping[str, float]:
        """Execute ngspice or equivalent backends and return metric snapshots."""

    @abstractmethod
    def optimize(self, context: ExecutionContext, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        """Iteratively update the circuit until targets converge."""

    def run(self, context: ExecutionContext) -> AgentResult:
        """End-to-end helper that mirrors the notebook pipeline."""

        plan_messages = self.build_plan(context)
        context.messages.messages.extend(plan_messages)
        tool_messages = self.select_tools(context, plan_messages)
        context.messages.messages.extend(tool_messages)
        metrics = self.run_simulation(context)
        optimized_metrics = self.optimize(context, metrics)
        return AgentResult(
            success=True,
            messages=context.messages.messages,
            artifacts={name: str(path) for name, path in context.artifacts.items()},
            metrics=optimized_metrics,
        )
