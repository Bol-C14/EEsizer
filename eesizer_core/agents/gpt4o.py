"""GPT-4o agent tuned for fast iterations while preserving notebook order."""
from __future__ import annotations

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import AgentMetadata
from .notebook import build_default_agent_config
from .simple import OptimizationTargets, SimpleSizingAgent


class Gpt4oAgent(SimpleSizingAgent):
    """GPT-4o agent that runs a leaner sizing loop."""

    metadata = AgentMetadata(
        name="gpt4o-agent",
        model_family="gpt",
        description="GPT-4o agent optimized for quick planning and simulation passes.",
        capabilities=("planning", "tool_selection", "simulation", "optimization"),
    )

    default_model = "gpt-4o"

    @classmethod
    def default_config(
        cls,
        simulation: SimulationConfig,
        optimization: OptimizationConfig,
        *,
        output_paths: OutputPathPolicy | None = None,
    ) -> AgentConfig:
        return build_default_agent_config(
            name="gpt4o",
            model=cls.default_model,
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            description="Fast-path GPT-4o sizing agent mirroring the notebook flow.",
            extra={
                "token_limit": 32_000,
                "retry": {"max_attempts": 3, "backoff_seconds": 1.5},
                "max_iterations": 10,
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


__all__ = ["Gpt4oAgent"]
