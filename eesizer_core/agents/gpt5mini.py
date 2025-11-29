"""GPT-5 Mini agent for lightweight sizing guidance."""
from __future__ import annotations

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import AgentMetadata
from .notebook import build_default_agent_config
from .simple import OptimizationTargets, SimpleSizingAgent


class Gpt5MiniAgent(SimpleSizingAgent):
    """Budget-friendly GPT-5 mini agent that reuses the shared simulation flow."""

    metadata = AgentMetadata(
        name="gpt5mini-agent",
        model_family="gpt",
        description="GPT-5 Mini agent focused on quick hints and reuse of shared metrics.",
        capabilities=("planning", "tool_selection", "simulation", "optimization"),
    )

    default_model = "gpt-5.1-mini"

    @classmethod
    def default_config(
        cls,
        simulation: SimulationConfig,
        optimization: OptimizationConfig,
        *,
        output_paths: OutputPathPolicy | None = None,
    ) -> AgentConfig:
        return build_default_agent_config(
            name="gpt5mini",
            model=cls.default_model,
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            description="Lightweight GPT-5 Mini agent with constrained budgets.",
            extra={
                "token_limit": 8_000,
                "retry": {"max_attempts": 2, "backoff_seconds": 1.0},
                "max_iterations": 6,
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


__all__ = ["Gpt5MiniAgent"]
