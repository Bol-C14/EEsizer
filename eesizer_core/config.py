"""Configuration scaffolding for agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(slots=True)
class SimulationConfig:
    """Settings shared by ngspice runners across agents."""

    binary_path: Path
    timeout_seconds: int = 120
    working_root: Path = Path("output")


@dataclass(slots=True)
class OptimizationConfig:
    """Stop criteria and tolerances for the sizing loop."""

    max_iterations: int = 25
    tolerance_percent: float = 0.05
    vgs_margin_volts: float = 0.05


@dataclass(slots=True)
class AgentConfig:
    """Top-level knobs for a single agent instance."""

    name: str
    model: str
    simulation: SimulationConfig
    optimization: OptimizationConfig
    extra: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestratorConfig:
    """Combines common defaults with per-agent overrides."""

    default_model: str
    simulation: SimulationConfig
    optimization: OptimizationConfig
    agents: MutableMapping[str, AgentConfig] = field(default_factory=dict)


class ConfigLoader:
    """Utility that transforms nested dictionaries into dataclasses."""

    def __init__(self, raw: Mapping[str, Any]):
        self.raw = raw

    def build_simulation(self, data: Mapping[str, Any] | None = None) -> SimulationConfig:
        data = data or self.raw.get("simulation", {})
        binary_path = Path(data.get("binary_path", "ngspice"))
        timeout = int(data.get("timeout_seconds", 120))
        work_root = Path(data.get("working_root", "output"))
        return SimulationConfig(binary_path=binary_path, timeout_seconds=timeout, working_root=work_root)

    def build_optimization(self, data: Mapping[str, Any] | None = None) -> OptimizationConfig:
        data = data or self.raw.get("optimization", {})
        return OptimizationConfig(
            max_iterations=int(data.get("max_iterations", 25)),
            tolerance_percent=float(data.get("tolerance_percent", 0.05)),
            vgs_margin_volts=float(data.get("vgs_margin_volts", 0.05)),
        )

    def build_agent(self, name: str, data: Mapping[str, Any]) -> AgentConfig:
        simulation = self.build_simulation(data.get("simulation"))
        optimization = self.build_optimization(data.get("optimization"))
        return AgentConfig(
            name=name,
            model=data.get("model", self.raw.get("default_model", "gpt-4o")),
            simulation=simulation,
            optimization=optimization,
            extra={k: v for k, v in data.items() if k not in {"model", "simulation", "optimization"}},
        )

    def build_orchestrator(self) -> OrchestratorConfig:
        simulation = self.build_simulation()
        optimization = self.build_optimization()
        agents_raw = self.raw.get("agents", {})
        agents: MutableMapping[str, AgentConfig] = {}
        for name, data in agents_raw.items():
            agents[name] = self.build_agent(name, data)
        return OrchestratorConfig(
            default_model=self.raw.get("default_model", "gpt-4o"),
            simulation=simulation,
            optimization=optimization,
            agents=agents,
        )
