"""Gemini 3.0 agent subclass aligned with the notebook blueprint."""
from __future__ import annotations

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import AgentMetadata
from .notebook import build_default_agent_config
from .simple import OptimizationTargets, SimpleSizingAgent


class Gemini30Agent(SimpleSizingAgent):
    """Gemini 3.0 agent that shares the common ngspice-backed simulation path."""

    metadata = AgentMetadata(
        name="gemini30-agent",
        model_family="gemini",
        description="Gemini 3.0 sizing agent with notebook-style tool sequencing.",
        capabilities=("planning", "tool_selection", "simulation", "optimization"),
    )

    default_model = "gemini-3.0-pro"

    @classmethod
    def default_config(
        cls,
        simulation: SimulationConfig,
        optimization: OptimizationConfig,
        *,
        output_paths: OutputPathPolicy | None = None,
    ) -> AgentConfig:
        return build_default_agent_config(
            name="gemini30",
            model=cls.default_model,
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            description="Gemini 3.0 agent aligned with legacy notebook prompts and outputs.",
            extra={
                "token_limit": 60_000,
                "retry": {"max_attempts": 3, "backoff_seconds": 2.5},
            },
        )

    def __init__(
        self,
        config: AgentConfig,
        simulator: NgSpiceRunner | MockNgSpiceSimulator,
        goal: str,
        targets: OptimizationTargets,
        *,
        tool_configs=None,
        recordings=None,
        provider=None,
    ) -> None:
        super().__init__(
            config,
            simulator,
            goal=goal,
            targets=targets,
            tool_configs=tool_configs,
            recordings=recordings,
            provider=provider,
        )


__all__ = ["Gemini30Agent"]
