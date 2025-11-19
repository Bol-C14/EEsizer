"""Reference implementation of the Agent lifecycle using the mock simulator."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from ..config import AgentConfig
from ..context import ArtifactKind, ExecutionContext
from ..messaging import Message, MessageRole, ToolCall, ToolResult
from ..netlist import NetlistSummary, summarize_netlist
from ..prompts import PromptLibrary
from ..simulation import MockNgSpiceSimulator
from ..spice import (
    ControlDeck,
    ac_simulation,
    describe_measurements,
    dc_simulation,
    measure_gain,
    measure_power,
    measure_voltage,
    tran_simulation,
)
from ..toolchain import ToolChainExecutor, ToolChainParser, ToolRegistry
from .base import Agent, AgentMetadata


@dataclass(slots=True)
class OptimizationTargets:
    gain_db: float
    power_mw: float


@dataclass(slots=True)
class OptimizationHistoryEntry:
    iteration: int
    gain_db: float
    power_mw: float
    analysis_note: str
    optimization_note: str
    sizing_note: str


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
        self.prompts = PromptLibrary()
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolChainExecutor(self.tool_registry)
        self.tool_parser = ToolChainParser()
        self._register_tooling()

    def build_plan(self, context: ExecutionContext) -> Sequence[Message]:
        """Emit a planning note that assumes ``context.netlist_path`` is set."""

        netlist_text = self._ensure_netlist_loaded(context)
        summary = summarize_netlist(context.netlist_path, netlist_text)
        self._summary = summary
        template = self.prompts.load("tasks_generation_template")
        content = template.render(goal=self.goal, netlist_summary=summary.describe())
        return [Message(role=MessageRole.SYSTEM, content=content, name=self.metadata.name)]

    def select_tools(self, context: ExecutionContext, history: Sequence[Message]) -> Sequence[Message]:
        """Describe the simulator/tooling choices used by downstream stages."""

        summary_desc = self._summary.describe() if self._summary else ""
        tool_plan = [
            {
                "name": "prepare_ac_plan",
                "arguments": {
                    "deck_name": "ota_ac_combo",
                    "output_node": "out",
                    "input_node": "in",
                    "supply_source": "vdd",
                },
            },
            {"name": "run_ngspice_simulation", "arguments": {"deck_ref": "prepare_ac_plan"}},
            {"name": "summarize_simulation_results", "arguments": {}},
        ]
        payload = json.dumps(tool_plan, indent=2)
        content = (
            f"Selecting tools for {self.metadata.name}. {summary_desc}\n"
            "Tool-chain blueprint:\n"
            f"```json\n{payload}\n```"
        )
        return [Message(role=MessageRole.USER, content=content, name="tool_selector")]

    def run_simulation(self, context: ExecutionContext) -> MutableMapping[str, float]:
        """Run the deterministic simulator and log telemetry in the execution context."""

        netlist_text = self._ensure_netlist_loaded(context)
        state: MutableMapping[str, Any] = {"netlist_text": netlist_text}
        tool_messages = context.messages.messages[-1:]
        parsed_calls = self.tool_parser.parse(tool_messages)
        if not parsed_calls:
            parsed_calls = [
                ToolCall(name="prepare_ac_plan", arguments={}, call_id="prepare_ac_plan"),
                ToolCall(
                    name="run_ngspice_simulation",
                    arguments={"deck_ref": "prepare_ac_plan"},
                    call_id="run_ngspice",
                ),
                ToolCall(name="summarize_simulation_results", arguments={}, call_id="summarize"),
            ]
        chain = self.tool_executor.run(parsed_calls, context, state=state)
        metrics = dict(chain.state.get("metrics", {}))
        if not metrics:
            metrics = self.simulator.run(netlist_text, None, context.working_dir)
        summary_text = chain.summary or describe_measurements(metrics)
        deck = chain.state.get("last_deck")
        if isinstance(deck, ControlDeck):
            template = self.prompts.load("simulation_function_explanation")
            analysis_list = ", ".join(d.label for d in deck.directives)
            measurement_list = ", ".join(m.name for m in deck.measurements)
            context.log(
                Message(
                    role=MessageRole.SYSTEM,
                    content=template.render(
                        deck_name=deck.name,
                        analysis_list=analysis_list,
                        measurement_list=measurement_list,
                        working_dir=str(context.working_dir),
                    ),
                    name="simulation_orchestrator",
                )
            )
        context.metadata["sim_output"] = summary_text
        context.log(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"Simulation complete. Metrics: {json.dumps(metrics, indent=2)}",
                name=self.metadata.name,
            )
        )
        metrics_path = context.working_dir / "simulation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        context.attach_artifact(
            "simulation_metrics",
            metrics_path,
            kind=ArtifactKind.SIMULATION,
            description="AC/DC sweep measurements produced by the mock simulator.",
        )
        summary_path = context.working_dir / "simulation_summary.txt"
        summary_path.write_text(summary_text + "\n")
        context.attach_artifact(
            "simulation_summary",
            summary_path,
            kind=ArtifactKind.SIMULATION,
            description="Derived sim_output summary mirroring the notebook log.",
        )
        return metrics

    def optimize(self, context: ExecutionContext, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        """Iteratively nudge metrics upward while mirroring the notebook log style."""

        optimized = dict(metrics)
        history: list[OptimizationHistoryEntry] = []
        max_iterations = self.config.optimization.max_iterations
        tolerance = self.config.optimization.tolerance_percent
        for iteration in range(1, max_iterations + 1):
            analysis_note = self.prompts.load("analysing_system_prompt").render(
                iteration=iteration,
                gain_db=optimized.get("gain_db", 0.0),
                power_mw=optimized.get("power_mw", 0.0),
                bandwidth_hz=optimized.get("bandwidth_hz", 0.0),
                transistor_count=optimized.get("transistor_count", 0.0),
            )
            context.log(
                Message(role=MessageRole.USER, content=analysis_note, name="analysis_prompt")
            )
            optimization_note = self.prompts.load("optimization_prompt").render(
                iteration=iteration,
                power_budget=self.targets.power_mw,
                target_gain_db=self.targets.gain_db,
            )
            context.log(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=optimization_note,
                    name=self.metadata.name,
                )
            )
            next_metrics = self._nudge_metrics(optimized)
            gain_delta = next_metrics.get("gain_db", 0.0) - optimized.get("gain_db", 0.0)
            power_delta = next_metrics.get("power_mw", 0.0) - optimized.get("power_mw", 0.0)
            sizing_note = self.prompts.load("sizing_prompt").render(
                iteration=iteration,
                gain_delta=gain_delta,
                power_delta=power_delta,
            )
            context.log(
                Message(role=MessageRole.ASSISTANT, content=sizing_note, name="sizing_logger")
            )
            optimized = next_metrics
            history.append(
                OptimizationHistoryEntry(
                    iteration=iteration,
                    gain_db=optimized.get("gain_db", 0.0),
                    power_mw=optimized.get("power_mw", 0.0),
                    analysis_note=analysis_note,
                    optimization_note=optimization_note,
                    sizing_note=sizing_note,
                )
            )
            if self._meets_targets(optimized, tolerance):
                break

        optimized["gain_db"] = max(optimized.get("gain_db", 0.0), self.targets.gain_db)
        optimized["power_mw"] = min(optimized.get("power_mw", self.targets.power_mw), self.targets.power_mw)
        optimized["iterations"] = float(len(history))
        optimized["meets_gain"] = float(optimized.get("gain_db", 0.0) >= self.targets.gain_db)
        optimized["meets_power"] = float(optimized.get("power_mw", 0.0) <= self.targets.power_mw)

        self._emit_target_summary(context)
        optimized_path = context.working_dir / "optimization_summary.json"
        optimized_path.write_text(json.dumps(optimized, indent=2))
        context.attach_artifact(
            "optimization_summary",
            optimized_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Notebook-style optimization summary containing iterations and flags.",
        )
        history_csv = context.working_dir / "optimization_history.csv"
        with history_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["iteration", "gain_db", "power_mw", "analysis_note"]
            )
            writer.writeheader()
            for entry in history:
                writer.writerow(
                    {
                        "iteration": entry.iteration,
                        "gain_db": f"{entry.gain_db:.3f}",
                        "power_mw": f"{entry.power_mw:.3f}",
                        "analysis_note": entry.analysis_note,
                    }
                )
        context.attach_artifact(
            "optimization_history_csv",
            history_csv,
            kind=ArtifactKind.OPTIMIZATION,
            description="Iteration-by-iteration metrics stored as CSV.",
        )
        history_log = context.working_dir / "optimization_history.log"
        history_text = []
        for entry in history:
            history_text.append(
                f"Iteration {entry.iteration}: gain={entry.gain_db:.3f} dB, power={entry.power_mw:.3f} mW\n"
                f"{entry.analysis_note}\n{entry.optimization_note}\n{entry.sizing_note}"
            )
        history_log.write_text("\n\n".join(history_text))
        context.attach_artifact(
            "optimization_history_log",
            history_log,
            kind=ArtifactKind.OPTIMIZATION,
            description="Text log summarizing prompts and adjustments per iteration.",
        )
        history_pdf = context.working_dir / "optimization_history.pdf"
        history_pdf.write_text(
            "Notebook-equivalent optimization report\n\n" + "\n\n".join(history_text)
        )
        context.attach_artifact(
            "optimization_history_pdf",
            history_pdf,
            kind=ArtifactKind.OPTIMIZATION,
            description="Lightweight PDF placeholder mirroring the notebook artifact list.",
        )

        copied_netlist = context.working_dir / context.netlist_path.name
        if not copied_netlist.exists():
            shutil.copy2(context.netlist_path, copied_netlist)
        context.attach_artifact(
            "netlist_copy",
            copied_netlist,
            kind=ArtifactKind.NETLIST,
            description="Input netlist snapshot captured for reproducibility.",
        )

        return optimized

    def _emit_target_summary(self, context: ExecutionContext) -> None:
        template = self.prompts.load("target_value_system_prompt")
        content = template.render(
            goal=self.goal,
            target_gain_db=self.targets.gain_db,
            target_power_mw=self.targets.power_mw,
            tolerance_percent=self.config.optimization.tolerance_percent,
        )
        context.log(Message(role=MessageRole.SYSTEM, content=content, name="spec_checker"))

    def _register_tooling(self) -> None:
        self.tool_registry.register("prepare_ac_plan", self._tool_prepare_ac_plan)
        self.tool_registry.register("run_ngspice_simulation", self._tool_run_simulation)
        self.tool_registry.register(
            "summarize_simulation_results", self._tool_summarize_results
        )

    def _tool_prepare_ac_plan(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        deck = self._build_default_deck(call.arguments)
        deck_key = self._deck_key(call.call_id or call.name)
        state[deck_key] = deck
        state[self._deck_key(call.name)] = deck
        state["last_deck"] = deck
        return ToolResult(
            call_id=call.call_id or deck.name,
            content={
                "deck_name": deck.name,
                "analyses": len(deck.directives),
                "measurements": len(deck.measurements),
            },
        )

    def _tool_run_simulation(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        deck_ref = call.arguments.get("deck_ref") or call.arguments.get("use")
        deck = None
        if deck_ref:
            deck = state.get(self._deck_key(deck_ref))
        if deck is None:
            deck = state.get("last_deck")
        if deck is None:
            deck = self._build_default_deck({})
        netlist_text = state.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValueError("Netlist text missing from tool-chain state")
        metrics = self.simulator.run(netlist_text, deck, context.working_dir)
        state["metrics"] = metrics
        return ToolResult(call_id=call.call_id or "run_ngspice", content=dict(metrics))

    def _tool_summarize_results(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        metrics = state.get("metrics", {})
        summary = describe_measurements(metrics if isinstance(metrics, Mapping) else {})
        state["simulation_summary"] = summary
        return ToolResult(call_id=call.call_id or "summary", content={"summary": summary})

    def _build_default_deck(self, arguments: Mapping[str, Any]) -> ControlDeck:
        deck_name = str(arguments.get("deck_name", "ota_ac_combo"))
        output_node = str(arguments.get("output_node", "out"))
        input_node = str(arguments.get("input_node", "in"))
        supply_source = str(arguments.get("supply_source", "vdd"))
        directives = (
            dc_simulation(
                source=str(arguments.get("dc_source", "VIN")),
                start=float(arguments.get("dc_start", -0.1)),
                stop=float(arguments.get("dc_stop", 0.1)),
                step=float(arguments.get("dc_step", 0.01)),
                description="DC operating point sweep",
            ),
            ac_simulation(
                sweep=str(arguments.get("ac_sweep", "dec")),
                points=int(arguments.get("ac_points", 40)),
                start_hz=float(arguments.get("ac_start", 10.0)),
                stop_hz=float(arguments.get("ac_stop", 1e6)),
                description="AC magnitude sweep",
            ),
            tran_simulation(
                step=float(arguments.get("tran_step", 1e-7)),
                stop=float(arguments.get("tran_stop", 1e-3)),
                description="Transient window for THD/power",
            ),
        )
        measurements = (
            measure_gain(
                "gain_db",
                output_node=output_node,
                input_node=input_node,
                description="AC gain measurement",
            ),
            measure_voltage(
                "vout_dc", node=output_node, description="Output DC bias measurement"
            ),
            measure_power(
                "power_mw",
                supply_source=supply_source,
                description="Average power consumption in mW",
            ),
        )
        return ControlDeck(name=deck_name, directives=directives, measurements=measurements)

    def _nudge_metrics(self, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        updated = dict(metrics)
        gain = updated.get("gain_db", 0.0)
        gain_delta = max(0.5, min(2.0, self.targets.gain_db - gain))
        updated["gain_db"] = gain + gain_delta * 0.5
        power = updated.get("power_mw", self.targets.power_mw)
        if power > self.targets.power_mw:
            power -= 0.05
        else:
            power = max(power - 0.01, self.targets.power_mw * 0.95)
        updated["power_mw"] = power
        bandwidth = updated.get("bandwidth_hz", 1e6)
        updated["bandwidth_hz"] = max(1e3, bandwidth * 0.99)
        updated.setdefault("transistor_count", 0.0)
        return updated

    def _meets_targets(self, metrics: Mapping[str, float], tolerance: float) -> bool:
        gain = metrics.get("gain_db", 0.0)
        power = metrics.get("power_mw", 0.0)
        gain_ok = gain >= self.targets.gain_db * (1 - tolerance)
        power_ok = power <= self.targets.power_mw * (1 + tolerance)
        return gain_ok and power_ok

    def _deck_key(self, identifier: str | None) -> str:
        return f"deck:{identifier or 'default'}"

    def _ensure_netlist_loaded(self, context: ExecutionContext) -> str:
        if not context.netlist_path:
            raise ValueError("ExecutionContext.netlist_path must be set before running the agent")
        if not context.netlist_path.exists():
            raise FileNotFoundError(context.netlist_path)
        return context.netlist_path.read_text()


__all__ = ["SimpleSizingAgent", "OptimizationTargets"]
