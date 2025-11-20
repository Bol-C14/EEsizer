"""Shared helpers for notebook-parity LLM agents."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

from ..config import AgentConfig, OptimizationConfig, OutputPathPolicy, SimulationConfig


def build_default_agent_config(
    name: str,
    model: str,
    simulation: SimulationConfig,
    optimization: OptimizationConfig,
    *,
    output_paths: OutputPathPolicy | None = None,
    description: str | None = None,
    tools: Mapping[str, str] | tuple[str, ...] | list[str] | None = None,
    prompt_overrides: Mapping[str, str] | None = None,
    prompt_paths: tuple[str, ...] | list[str] | None = None,
    extra: MutableMapping[str, object] | None = None,
) -> AgentConfig:
    """Helper to assemble an AgentConfig with consistent defaults.

    The helper mirrors notebook assumptions: ngspice as the primary tool,
    rich token budgets, and retry/backoff policies stored in ``extra`` for
    orchestrator/transport layers to honor.
    """

    tool_list: tuple[str, ...]
    if tools is None:
        tool_list = ("ngspice",)
    elif isinstance(tools, Mapping):
        tool_list = tuple(tools.values())
    else:
        tool_list = tuple(tools)

    return AgentConfig(
        name=name,
        model=model,
        simulation=simulation,
        optimization=optimization,
        tools=tool_list,
        description=description,
        output_paths=output_paths,
        prompt_overrides=dict(prompt_overrides or {}),
        prompt_paths=tuple(Path(path) for path in (prompt_paths or ())),
        extra=dict(extra or {}),
    )


__all__ = ["build_default_agent_config"]
