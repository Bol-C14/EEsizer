"""Reference implementation of the Agent lifecycle using the mock simulator."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from ..config import AgentConfig
from ..context import ExecutionContext
from ..messaging import Message, MessageRole
from ..netlist import NetlistSummary, summarize_netlist
from ..simulation import MockNgSpiceSimulator
from .base import Agent, AgentMetadata


@dataclass(slots=True)
class OptimizationTargets:
    gain_db: float
    power_mw: float


class SimpleSizingAgent(Agent):
    """Minimal-yet-testable agent that orchestrates the sizing loop."""

    metadata = AgentMetadata(
        name="simple-sizing-agent",
        model_family="gpt",
        description="Reference agent that migrates notebook logic into Python",
        capabilities=("planning", "simulation", "optimization"),
    )

    def __init__(
        self,
        config: AgentConfig,
        simulator: MockNgSpiceSimulator,
        goal: str,
        targets: OptimizationTargets,
    ) -> None:
        super().__init__(config)
        self.simulator = simulator
        self.goal = goal
        self.targets = targets
        self._summary: NetlistSummary | None = None

    def build_plan(self, context: ExecutionContext) -> Sequence[Message]:
        netlist_text = self._ensure_netlist_loaded(context)
        summary = summarize_netlist(context.netlist_path, netlist_text)
        self._summary = summary
        content = (
            f"Goal: {self.goal}\n"
            f"Summary: {summary.describe()}\n"
            f"Plan: 1) analyze topology 2) run mock simulation 3) optimize to gain>={self.targets.gain_db} dB"
        )
        return [Message(role=MessageRole.SYSTEM, content=content, name=self.metadata.name)]

    def select_tools(self, context: ExecutionContext, history: Sequence[Message]) -> Sequence[Message]:
        summary_desc = self._summary.describe() if self._summary else ""
        content = (
            f"Selecting tools for {self.metadata.name}. {summary_desc}\n"
            "Using MockNgSpiceSimulator with AC + DC sweeps and saving metrics to JSON."
        )
        return [Message(role=MessageRole.USER, content=content, name="tool_selector")]

    def run_simulation(self, context: ExecutionContext) -> MutableMapping[str, float]:
        netlist_text = self._ensure_netlist_loaded(context)
        metrics = self.simulator.run(netlist_text)
        context.log(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"Simulation complete. Metrics: {json.dumps(metrics, indent=2)}",
                name=self.metadata.name,
            )
        )
        metrics_path = context.working_dir / "simulation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        context.attach_artifact("simulation_metrics", metrics_path)
        return metrics

    def optimize(self, context: ExecutionContext, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        optimized = dict(metrics)
        iterations = 0
        while optimized.get("gain_db", 0.0) < self.targets.gain_db and iterations < self.config.optimization.max_iterations:
            optimized["gain_db"] += 1.5
            iterations += 1
        if optimized.get("power_mw", 0.0) > self.targets.power_mw:
            optimized["power_mw"] = max(self.targets.power_mw, optimized["power_mw"] - 0.1)
        optimized["iterations"] = float(iterations)
        optimized["meets_gain"] = float(optimized["gain_db"] >= self.targets.gain_db)
        optimized["meets_power"] = float(optimized["power_mw"] <= self.targets.power_mw)

        optimized_path = context.working_dir / "optimization_summary.json"
        optimized_path.write_text(json.dumps(optimized, indent=2))
        context.attach_artifact("optimization_summary", optimized_path)
        context.log(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"Optimization iterations: {iterations}. Final metrics: {json.dumps(optimized, indent=2)}",
                name=self.metadata.name,
            )
        )

        copied_netlist = context.working_dir / context.netlist_path.name
        if not copied_netlist.exists():
            shutil.copy2(context.netlist_path, copied_netlist)
        context.attach_artifact("netlist_copy", copied_netlist)

        return optimized

    def _ensure_netlist_loaded(self, context: ExecutionContext) -> str:
        if not context.netlist_path:
            raise ValueError("ExecutionContext.netlist_path must be set before running the agent")
        if not context.netlist_path.exists():
            raise FileNotFoundError(context.netlist_path)
        return context.netlist_path.read_text()


__all__ = ["SimpleSizingAgent", "OptimizationTargets"]
