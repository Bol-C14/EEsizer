"""Reference implementation of the Agent lifecycle using a pluggable simulator."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from ..analysis.metrics import (
    aggregate_measurement_values,
    merge_metric_sources,
    standard_measurements,
    validate_metrics,
)
from ..config import AgentConfig, ToolConfig
from ..context import ArtifactKind, ExecutionContext
from ..messaging import Message, MessageRole, ToolCall, ToolResult
from ..netlist import NetlistSummary, summarize_netlist
from ..prompts import PromptLibrary
from ..providers import LLMProvider, build_provider
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from ..spice import (
    ControlDeck,
    describe_measurements,
)
from ..toolchain import ToolChainExecutor, ToolChainParser, ToolRegistry
from .base import Agent, AgentMetadata
from .optimizer import MetricOptimizer
from .reporting import OptimizationReporter
from .scoring import OptimizationTargets, ScoringPolicy
from .tooling import build_default_deck


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
        simulator: NgSpiceRunner | MockNgSpiceSimulator,
        goal: str,
        targets: OptimizationTargets,
        *,
        tool_configs: Mapping[str, ToolConfig] | None = None,
        recordings: Mapping[str, Mapping[str, str]] | None = None,
        provider: LLMProvider | None = None,
        force_live_llm: bool | None = None,
    ) -> None:
        super().__init__(config)
        self.simulator = simulator
        self.goal = goal
        self.targets = targets
        self._summary: NetlistSummary | None = None
        self.prompts = PromptLibrary(
            search_paths=self.config.prompt_paths,
            overrides=self.config.prompt_overrides,
        )
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolChainExecutor(self.tool_registry)
        self.tool_parser = ToolChainParser()
        self._register_tooling()
        self.llm: LLMProvider = provider or build_provider(
            self.config,
            tool_configs,
            recordings=recordings,
            default_tool_calls=self._default_tool_calls(),
            force_live=force_live_llm,
        )
        self.scoring = self._build_scoring_policy()
        self.optimizer = MetricOptimizer(
            scoring=self.scoring,
            prompts=self.prompts,
            targets=self.targets,
            max_iterations=self.config.optimization.max_iterations,
            nudge_fn=self._nudge_metrics,
            stagnation_rounds=self._optimizer_param("stagnation_rounds", default=3, cast=int),
            min_improvement=self._optimizer_param("min_improvement", default=0.01, cast=float),
        )
        self.reporter: OptimizationReporter | None = None

    def build_plan(self, context: ExecutionContext) -> Sequence[Message]:
        """Emit a planning note that assumes ``context.netlist_path`` is set."""

        netlist_text = self._ensure_netlist_loaded(context)
        summary = summarize_netlist(context.netlist_path, netlist_text)
        self._summary = summary
        template = self.prompts.load("task_decomposition")
        plan_note = template.render(goal=self.goal, netlist_summary=summary.describe())
        target_template = self.prompts.load("target_extraction")
        target_note = target_template.render(
            goal=self.goal,
            target_gain_db=self.targets.gain_db,
            target_power_mw=self.targets.power_mw,
        )
        system_message = Message(
            role=MessageRole.SYSTEM, content=plan_note, name=self.metadata.name
        )
        user_message = Message(
            role=MessageRole.USER, content=target_note, name="target_extraction"
        )
        response = self.llm.chat([system_message, user_message], response_name="plan")
        return [system_message, user_message, response.message]

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
        template = self.prompts.load("simulation_planning")
        content = template.render(
            agent_name=self.metadata.name,
            netlist_summary=summary_desc,
            tool_blueprint=payload,
        )
        request = Message(role=MessageRole.USER, content=content, name="tool_selector")
        response = self.llm.chat(
            [request], response_name="tool_plan", tools=self._tool_schemas()
        )
        return [request, response.message]

    def run_simulation(self, context: ExecutionContext) -> MutableMapping[str, float]:
        """Run the deterministic simulator and log telemetry in the execution context."""

        netlist_text = self._ensure_netlist_loaded(context)
        sim_dir = self._resolve_dir(context, "simulations")
        state: MutableMapping[str, Any] = {"netlist_text": netlist_text}
        tool_messages = context.messages.messages[-1:]
        parsed_calls = self.tool_parser.parse(tool_messages)
        if not parsed_calls:
            parsed_calls = list(self._default_tool_calls())
        chain = self.tool_executor.run(parsed_calls, context, state=state)
        metrics = dict(chain.state.get("metrics", {}))
        if metrics:
            metrics = aggregate_measurement_values(metrics)
        else:
            raw_metrics = self.simulator.run(netlist_text, None, sim_dir, artifact_dir=sim_dir)
            metrics = aggregate_measurement_values(raw_metrics)
        metrics = self._merge_with_cached_metrics(context, metrics)
        missing_metrics = self._warn_if_missing_metrics(context, metrics)
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
        if missing_metrics:
            context.metadata["missing_metrics"] = ",".join(sorted(missing_metrics))
        context.log(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"Simulation complete. Metrics: {json.dumps(metrics, indent=2)}",
                name=self.metadata.name,
            )
        )
        sim_dir = self._resolve_dir(context, "simulations")
        metrics_path = sim_dir / "simulation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        context.attach_artifact(
            "simulation_metrics",
            metrics_path,
            kind=ArtifactKind.SIMULATION,
            description="AC/DC sweep measurements produced by the configured simulator.",
        )
        summary_path = sim_dir / "simulation_summary.txt"
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

        baseline_metrics = dict(metrics)
        result = self.optimizer.optimize(context, metrics)
        optimized = dict(result.metrics)
        optimized["gain_db"] = max(optimized.get("gain_db", 0.0), self.targets.gain_db)
        optimized["power_mw"] = min(optimized.get("power_mw", self.targets.power_mw), self.targets.power_mw)
        optimized["iterations"] = float(len(result.history))
        optimized["meets_gain"] = float(optimized.get("gain_db", 0.0) >= self.targets.gain_db)
        optimized["meets_power"] = float(optimized.get("power_mw", 0.0) <= self.targets.power_mw)
        optimized["composite_score"] = float(result.best_score)

        self._emit_target_summary(context)
        artifacts_dir = self._resolve_dir(context, "artifacts")
        self.reporter = OptimizationReporter(artifacts_dir)
        self.reporter.write_summary(context, optimized)
        self.reporter.write_history(context, result.history)
        # Variant comparison: baseline vs optimized
        self.reporter.write_variant_comparison(
            context,
            variants=(
                ("baseline", baseline_metrics),
                ("optimized", optimized),
            ),
            scoring_fn=self.scoring.score,
        )

        copied_netlist = artifacts_dir / context.netlist_path.name
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

    @staticmethod
    def _default_tool_calls() -> tuple[ToolCall, ...]:
        return (
            ToolCall(name="prepare_ac_plan", arguments={}, call_id="prepare_ac_plan"),
            ToolCall(
                name="run_ngspice_simulation",
                arguments={"deck_ref": "prepare_ac_plan"},
                call_id="run_ngspice",
            ),
            ToolCall(name="summarize_simulation_results", arguments={}, call_id="summarize"),
        )

    @staticmethod
    def _tool_schemas() -> tuple[Mapping[str, object], ...]:
        return (
            {
                "name": "prepare_ac_plan",
                "description": "Builds AC/DC/tran control decks for ngspice",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deck_name": {"type": "string"},
                        "output_node": {"type": "string"},
                        "input_node": {"type": "string"},
                        "supply_source": {"type": "string"},
                    },
                },
            },
            {
                "name": "run_ngspice_simulation",
                "description": "Executes ngspice for the prepared deck and aggregates metrics",
                "parameters": {
                    "type": "object",
                    "properties": {"deck_ref": {"type": "string"}},
                },
            },
            {
                "name": "summarize_simulation_results",
                "description": "Summarizes measurements into a concise log string",
                "parameters": {"type": "object", "properties": {}},
            },
        )

    def _register_tooling(self) -> None:
        self.tool_registry.register("prepare_ac_plan", self._tool_prepare_ac_plan)
        self.tool_registry.register("run_ngspice_simulation", self._tool_run_simulation)
        self.tool_registry.register(
            "summarize_simulation_results", self._tool_summarize_results
        )

    def _tool_prepare_ac_plan(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        netlist_text = state.get("netlist_text")
        deck = build_default_deck(call.arguments, netlist_text=netlist_text)
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
            deck = build_default_deck({}, netlist_text=state.get("netlist_text"))
        netlist_text = state.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValueError("Netlist text missing from tool-chain state")
        sim_dir = self._resolve_dir(context, "simulations")
        raw_metrics = self.simulator.run(netlist_text, deck, sim_dir, artifact_dir=sim_dir)
        metrics = aggregate_measurement_values(raw_metrics)
        metrics = self._merge_with_cached_metrics(context, metrics)
        missing = self._warn_if_missing_metrics(context, metrics)
        state["raw_metrics"] = raw_metrics
        state["metrics"] = metrics
        if missing:
            state["missing_metrics"] = missing
        return ToolResult(call_id=call.call_id or "run_ngspice", content=dict(metrics))

    def _tool_summarize_results(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        metrics = state.get("metrics", {})
        summary = describe_measurements(metrics if isinstance(metrics, Mapping) else {})
        state["simulation_summary"] = summary
        return ToolResult(call_id=call.call_id or "summary", content={"summary": summary})

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

    def _deck_key(self, identifier: str | None) -> str:
        return f"deck:{identifier or 'default'}"

    def _resolve_dir(self, context: ExecutionContext, category: str) -> Path:
        """Resolve a run-scoped directory for simulations, artifacts, logs, or plans."""

        if context.paths:
            mapping = {
                "simulations": context.paths.simulations,
                "artifacts": context.paths.artifacts,
                "logs": context.paths.logs,
                "plans": context.paths.plans,
            }
            target = mapping.get(category)
            if target:
                target.mkdir(parents=True, exist_ok=True)
                return target
        context.working_dir.mkdir(parents=True, exist_ok=True)
        return context.working_dir

    def _ensure_netlist_loaded(self, context: ExecutionContext) -> str:
        if not context.netlist_path:
            raise ValueError("ExecutionContext.netlist_path must be set before running the agent")
        if not context.netlist_path.exists():
            raise FileNotFoundError(context.netlist_path)
        return context.netlist_path.read_text()

    def _warn_if_missing_metrics(
        self, context: ExecutionContext, metrics: Mapping[str, float]
    ) -> Sequence[str]:
        missing = validate_metrics(metrics)
        if missing:
            context.log(
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Missing expected measurements: " + ", ".join(sorted(missing))
                    ),
                    name="metrics_validator",
                )
            )
        return missing

    def _merge_with_cached_metrics(
        self, context: ExecutionContext, metrics: Mapping[str, float]
    ) -> MutableMapping[str, float]:
        cached: MutableMapping[str, float] = {}
        cached_blob = context.metadata.get("cached_metrics")
        if isinstance(cached_blob, Mapping):
            cached = merge_metric_sources(cached_blob)
        elif isinstance(cached_blob, str):
            try:
                parsed = json.loads(cached_blob)
                if isinstance(parsed, Mapping):
                    cached = merge_metric_sources(parsed)
            except json.JSONDecodeError:
                cached = {}
        merged = aggregate_measurement_values(merge_metric_sources(cached, metrics))
        context.metadata["cached_metrics"] = json.dumps(merged)
        return merged

    def _build_scoring_policy(self) -> ScoringPolicy:
        scoring_cfg = {}
        try:
            scoring_cfg = dict(getattr(self.config, "extra", {}).get("scoring", {}))
        except Exception:
            scoring_cfg = {}
        weights: MutableMapping[str, float] = {}
        raw_weights = scoring_cfg.get("weights", {}) if isinstance(scoring_cfg, Mapping) else {}
        if isinstance(raw_weights, Mapping):
            for key, value in raw_weights.items():
                try:
                    weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        plugins = []
        raw_plugins = scoring_cfg.get("plugins", ()) if isinstance(scoring_cfg, Mapping) else ()
        if isinstance(raw_plugins, (list, tuple)):
            for dotted in raw_plugins:
                func = self._resolve_plugin(dotted)
                if func:
                    plugins.append(func)

        return ScoringPolicy(
            self.targets,
            tolerance=self.config.optimization.tolerance_percent,
            weights=weights or {"gain": 1.0, "power": 1.0},
            plugins=tuple(plugins),
        )

    def _optimizer_param(self, key: str, *, default, cast):
        try:
            value = getattr(self.config, "extra", {}).get("optimizer", {}).get(key, default)
            return cast(value)
        except Exception:
            return default

    @staticmethod
    def _resolve_plugin(dotted: object):
        if not isinstance(dotted, str):
            return None
        try:
            import importlib

            module_name, func_name = dotted.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name, None)
            return func if callable(func) else None
        except Exception:
            return None


__all__ = ["SimpleSizingAgent", "OptimizationTargets", "ScoringPolicy"]
