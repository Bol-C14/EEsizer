"""Claude 3.5 agent subclass mirroring notebook ordering and prompts."""
from __future__ import annotations

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import AgentMetadata
from .notebook import build_default_agent_config
from .scoring import OptimizationTargets
from .simple import SimpleSizingAgent


class Claude35Agent(SimpleSizingAgent):
    """Anthropic Claude 3.5 agent wired to the shared ngspice tool-chain."""

    metadata = AgentMetadata(
        name="claude35-agent",
        model_family="claude",
        description="Claude 3.5 agent matching the notebook sequencing and artifacts.",
        capabilities=("planning", "tool_selection", "simulation", "optimization"),
    )

    default_model = "claude-3.5-sonnet"

    @classmethod
    def default_config(
        cls,
        simulation: SimulationConfig,
        optimization: OptimizationConfig,
        *,
        output_paths: OutputPathPolicy | None = None,
    ) -> AgentConfig:
        return build_default_agent_config(
            name="claude35",
            model=cls.default_model,
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            description="Claude 3.5 sizing agent aligned with notebook prompts.",
            extra={
                "token_limit": 200_000,
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
        force_live_llm: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            simulator,
            goal=goal,
            targets=targets,
            tool_configs=tool_configs,
            recordings=recordings,
            provider=provider,
            force_live_llm=force_live_llm,
        )


__all__ = ["Claude35Agent"]
