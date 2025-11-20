"""GPT-5 agent mirroring the notebook call order with ngspice tooling."""
from __future__ import annotations

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import AgentMetadata
from .notebook import build_default_agent_config
from .simple import OptimizationTargets, SimpleSizingAgent


class Gpt5Agent(SimpleSizingAgent):
    """High-capacity GPT-5 agent that follows the notebook lifecycle."""

    metadata = AgentMetadata(
        name="gpt5-agent",
        model_family="gpt",
        description="GPT-5 sizing agent aligned with notebook prompts and tool calls.",
        capabilities=("planning", "tool_selection", "simulation", "optimization"),
    )

    default_model = "gpt-5.1"

    @classmethod
    def default_config(
        cls,
        simulation: SimulationConfig,
        optimization: OptimizationConfig,
        *,
        output_paths: OutputPathPolicy | None = None,
    ) -> AgentConfig:
        return build_default_agent_config(
            name="gpt5",
            model=cls.default_model,
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            description="Notebook-derived GPT-5 sizing agent with generous context window.",
            extra={
                "token_limit": 120_000,
                "retry": {"max_attempts": 4, "backoff_seconds": 2.0},
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


__all__ = ["Gpt5Agent"]
