"""Configuration scaffolding for agents and orchestrators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class RunPathLayout:
    """Resolved set of paths for a single run.

    The layout nests artifacts under a run-specific directory and further under a
    netlist identifier so that notebook outputs (e.g. ``output/90nm/netlist_cs_o3``)
    are easy to mirror and compare.
    """

    run_dir: Path
    artifacts: Path
    logs: Path
    simulations: Path
    plans: Path


@dataclass(slots=True)
class OutputPathPolicy:
    """Describes how artifacts, logs, and plans should be organized on disk."""

    root: Path
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    simulations_dir: str = "simulations"
    plans_dir: str = "plans"

    def validate(self) -> None:
        if not str(self.root):  # Empty string check while allowing relative paths.
            raise ValueError("OutputPathPolicy.root must be a non-empty path")
        for field_name in ("artifacts_dir", "logs_dir", "simulations_dir", "plans_dir"):
            value = getattr(self, field_name)
            if not value:
                raise ValueError(f"{field_name} must be a non-empty string")

    def ensure_base_dir(self) -> Path:
        """Create the base directory for runs if missing."""

        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def ensure_run_structure(self, run_id: str) -> Path:
        """Create the run directory plus common subfolders."""

        base = self.ensure_base_dir()
        run_dir = base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for child in (self.artifacts_dir, self.logs_dir, self.simulations_dir, self.plans_dir):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        return run_dir

    def build_layout(self, run_id: str, *, netlist_stem: str | None = None) -> RunPathLayout:
        """Create a nested directory layout for a specific run/netlist."""

        run_dir = self.ensure_run_structure(run_id)
        suffix = netlist_stem or run_id

        def ensure_child(child: str) -> Path:
            path = run_dir / child / suffix
            path.mkdir(parents=True, exist_ok=True)
            return path

        return RunPathLayout(
            run_dir=run_dir,
            artifacts=ensure_child(self.artifacts_dir),
            logs=ensure_child(self.logs_dir),
            simulations=ensure_child(self.simulations_dir),
            plans=ensure_child(self.plans_dir),
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "root": str(self.root),
            "artifacts_dir": self.artifacts_dir,
            "logs_dir": self.logs_dir,
            "simulations_dir": self.simulations_dir,
            "plans_dir": self.plans_dir,
        }


@dataclass(slots=True)
class SimulationConfig:
    """Settings shared by ngspice runners across agents."""

    binary_path: Path
    timeout_seconds: int = 120

    def validate(self) -> None:
        if not str(self.binary_path):
            raise ValueError("SimulationConfig.binary_path must be provided")
        if self.timeout_seconds <= 0:
            raise ValueError("SimulationConfig.timeout_seconds must be positive")

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "binary_path": str(self.binary_path),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(slots=True)
class OptimizationConfig:
    """Stop criteria and tolerances for the sizing loop."""

    max_iterations: int = 25
    tolerance_percent: float = 0.05
    vgs_margin_volts: float = 0.05

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("OptimizationConfig.max_iterations must be positive")
        if self.tolerance_percent <= 0:
            raise ValueError("OptimizationConfig.tolerance_percent must be positive")
        if self.vgs_margin_volts < 0:
            raise ValueError("OptimizationConfig.vgs_margin_volts must be non-negative")

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "tolerance_percent": self.tolerance_percent,
            "vgs_margin_volts": self.vgs_margin_volts,
        }


@dataclass(slots=True)
class ToolConfig:
    """Describes credentials and parameters for an external tool (LLM, simulator, etc.)."""

    name: str
    kind: str
    credentials: MutableMapping[str, Any] = field(default_factory=dict)
    parameters: MutableMapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ToolConfig.name must be provided")
        if not self.kind:
            raise ValueError(f"ToolConfig.kind must be provided for tool '{self.name}'")

    def to_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "credentials": dict(self.credentials),
            "parameters": dict(self.parameters),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(slots=True)
class AgentConfig:
    """Top-level knobs for a single agent instance."""

    name: str
    model: str
    simulation: SimulationConfig
    optimization: OptimizationConfig
    tools: Sequence[str] = field(default_factory=tuple)
    description: str | None = None
    output_paths: OutputPathPolicy | None = None
    prompt_overrides: MutableMapping[str, str] = field(default_factory=dict)
    prompt_paths: Sequence[Path] = field(default_factory=tuple)
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("AgentConfig.name must be provided")
        if not self.model:
            raise ValueError("AgentConfig.model must be provided")
        self.simulation.validate()
        self.optimization.validate()
        if self.output_paths:
            self.output_paths.validate()

    def resolve_output_paths(self, default: OutputPathPolicy) -> OutputPathPolicy:
        return self.output_paths or default

    def to_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "model": self.model,
            "simulation": self.simulation.to_dict(),
            "optimization": self.optimization.to_dict(),
            "tools": list(self.tools),
            "prompt_overrides": dict(self.prompt_overrides),
            "prompt_paths": [str(path) for path in self.prompt_paths],
            "extra": dict(self.extra),
        }
        if self.description:
            payload["description"] = self.description
        if self.output_paths:
            payload["output_paths"] = self.output_paths.to_dict()
        return payload


@dataclass(slots=True)
class OrchestratorConfig:
    """Combines common defaults with per-agent overrides."""

    default_model: str
    simulation: SimulationConfig
    optimization: OptimizationConfig
    output_paths: OutputPathPolicy
    tools: MutableMapping[str, ToolConfig] = field(default_factory=dict)
    agents: MutableMapping[str, AgentConfig] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.default_model:
            raise ValueError("OrchestratorConfig.default_model must be provided")
        self.simulation.validate()
        self.optimization.validate()
        self.output_paths.validate()
        for tool in self.tools.values():
            tool.validate()
        if not self.agents:
            raise ValueError("At least one agent configuration must be defined")
        for agent in self.agents.values():
            agent.validate()
            for tool_name in agent.tools:
                if tool_name not in self.tools:
                    raise ValueError(
                        f"Agent '{agent.name}' references unknown tool '{tool_name}'."
                    )

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "default_model": self.default_model,
            "simulation": self.simulation.to_dict(),
            "optimization": self.optimization.to_dict(),
            "output_paths": self.output_paths.to_dict(),
            "tools": {name: cfg.to_dict() for name, cfg in self.tools.items()},
            "agents": {name: cfg.to_dict() for name, cfg in self.agents.items()},
        }


class ConfigLoader:
    """Utility that transforms nested dictionaries into dataclasses."""

    def __init__(self, raw: Mapping[str, Any]):
        self.raw = raw

    @staticmethod
    def _as_mapping(data: Any, label: str) -> Mapping[str, Any]:
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{label} must be a mapping, got {type(data)!r}")
        return data

    def build_simulation(self, data: Mapping[str, Any] | None = None) -> SimulationConfig:
        payload = self._as_mapping(data if data is not None else self.raw.get("simulation", {}), "simulation")
        config = SimulationConfig(
            binary_path=Path(payload.get("binary_path", "ngspice")),
            timeout_seconds=int(payload.get("timeout_seconds", 120)),
        )
        config.validate()
        return config

    def build_optimization(self, data: Mapping[str, Any] | None = None) -> OptimizationConfig:
        payload = self._as_mapping(data if data is not None else self.raw.get("optimization", {}), "optimization")
        config = OptimizationConfig(
            max_iterations=int(payload.get("max_iterations", 25)),
            tolerance_percent=float(payload.get("tolerance_percent", 0.05)),
            vgs_margin_volts=float(payload.get("vgs_margin_volts", 0.05)),
        )
        config.validate()
        return config

    def build_output_paths(self, data: Mapping[str, Any] | None = None) -> OutputPathPolicy:
        payload = self._as_mapping(data if data is not None else self.raw.get("output_paths", {}), "output_paths")
        policy = OutputPathPolicy(
            root=Path(payload.get("root", "output")),
            artifacts_dir=payload.get("artifacts_dir", "artifacts"),
            logs_dir=payload.get("logs_dir", "logs"),
            simulations_dir=payload.get("simulations_dir", "simulations"),
            plans_dir=payload.get("plans_dir", "plans"),
        )
        policy.validate()
        return policy

    def build_tool(self, name: str, data: Mapping[str, Any]) -> ToolConfig:
        payload = self._as_mapping(data, f"tools.{name}")
        kind = payload.get("kind") or payload.get("type")
        tool = ToolConfig(
            name=name,
            kind=str(kind or ""),
            credentials=dict(self._as_mapping(payload.get("credentials"), f"tools.{name}.credentials")),
            parameters=dict(self._as_mapping(payload.get("parameters"), f"tools.{name}.parameters")),
            description=payload.get("description"),
        )
        tool.validate()
        return tool

    def build_agent(self, name: str, data: Mapping[str, Any]) -> AgentConfig:
        payload = self._as_mapping(data, f"agents.{name}")
        simulation = self.build_simulation(payload.get("simulation"))
        optimization = self.build_optimization(payload.get("optimization"))
        output_paths = None
        if "output_paths" in payload:
            output_paths = self.build_output_paths(payload.get("output_paths"))
        tools_field = payload.get("tools", ())
        if isinstance(tools_field, str):
            tools: Sequence[str] = (tools_field,)
        else:
            tools = tuple(tools_field)
        prompt_paths_field = payload.get("prompt_paths", ())
        prompt_paths: Sequence[Path] = tuple(Path(p) for p in prompt_paths_field)
        agent = AgentConfig(
            name=name,
            model=payload.get("model", self.raw.get("default_model", "gpt-4o")),
            simulation=simulation,
            optimization=optimization,
            tools=tools,
            description=payload.get("description"),
            output_paths=output_paths,
            prompt_overrides=dict(payload.get("prompt_overrides", {})),
            prompt_paths=prompt_paths,
            extra={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "model",
                    "simulation",
                    "optimization",
                    "tools",
                    "description",
                    "output_paths",
                    "prompt_overrides",
                    "prompt_paths",
                }
            },
        )
        agent.validate()
        return agent

    def build_orchestrator(self) -> OrchestratorConfig:
        simulation = self.build_simulation()
        optimization = self.build_optimization()
        output_paths = self.build_output_paths()
        tools_raw = self._as_mapping(self.raw.get("tools", {}), "tools")
        tools: MutableMapping[str, ToolConfig] = {
            name: self.build_tool(name, data) for name, data in tools_raw.items()
        }
        agents_raw = self._as_mapping(self.raw.get("agents", {}), "agents")
        agents: MutableMapping[str, AgentConfig] = {
            name: self.build_agent(name, data) for name, data in agents_raw.items()
        }
        config = OrchestratorConfig(
            default_model=self.raw.get("default_model", "gpt-4o"),
            simulation=simulation,
            optimization=optimization,
            output_paths=output_paths,
            tools=tools,
            agents=agents,
        )
        config.validate()
        return config


__all__ = [
    "AgentConfig",
    "ConfigLoader",
    "OptimizationConfig",
    "OrchestratorConfig",
    "RunPathLayout",
    "OutputPathPolicy",
    "SimulationConfig",
    "ToolConfig",
]
