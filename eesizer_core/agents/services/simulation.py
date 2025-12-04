"""Simulation service that encapsulates tool execution and metric aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from ...analysis.metrics import aggregate_measurement_values, merge_metric_sources, validate_metrics
from ...context import ArtifactKind, ExecutionContext
from ...messaging import Message, MessageRole, ToolCall, ToolResult
from ...netlist_patch import ParamChange, apply_param_changes
from ...prompts import PromptLibrary
from ...simulation import MockNgSpiceSimulator, NgSpiceRunner
from ...spice import ControlDeck, describe_measurements
from ...toolchain import ToolChainExecutor, ToolChainParser, ToolRegistry
from ..tooling import build_default_deck

logger = logging.getLogger(__name__)


class SimulationService:
    """Runs netlist simulations via registered tool chains."""

    def __init__(
        self,
        simulator: NgSpiceRunner | MockNgSpiceSimulator,
        prompts: PromptLibrary,
        *,
        default_tool_calls: Sequence[ToolCall],
        tool_schemas: Sequence[Mapping[str, object]],
        failure_limit: int = 2,
    ) -> None:
        self.simulator = simulator
        self.prompts = prompts
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolChainExecutor(self.tool_registry)
        self.tool_parser = ToolChainParser()
        self.default_tool_calls = tuple(default_tool_calls)
        self.tool_schemas = tuple(tool_schemas)
        self._last_successful_metrics: MutableMapping[str, float] = {}
        self._failure_limit = max(1, int(failure_limit))
        self._register_tooling()

    def run(self, context: ExecutionContext) -> MutableMapping[str, float]:
        """Execute the tool chain and return aggregated metrics."""

        netlist_text = self._ensure_netlist_loaded(context)
        patched_netlist = self._apply_structured_changes(netlist_text, context)
        sim_dir = self._resolve_dir(context, "simulations")
        working_netlist = sim_dir / (context.netlist_path.name if context.netlist_path else "circuit.cir")
        try:
            working_netlist.write_text(patched_netlist)
            context.metadata["working_netlist_path"] = str(working_netlist)
        except OSError:
            pass

        state: MutableMapping[str, Any] = {"netlist_text": patched_netlist}
        tool_messages = context.messages.messages[-1:]
        parsed_calls = self.tool_parser.parse(tool_messages)
        if not parsed_calls:
            parsed_calls = list(self.default_tool_calls)

        try:
            chain = self.tool_executor.run(parsed_calls, context, state=state)
            metrics = dict(chain.state.get("metrics", {}))
            if metrics:
                metrics = aggregate_measurement_values(metrics)
            else:
                raw_metrics = self.simulator.run(patched_netlist, None, sim_dir, artifact_dir=sim_dir)
                metrics = aggregate_measurement_values(raw_metrics)
            self._last_successful_metrics = dict(metrics)
            context.metadata.pop("sim_failure_count", None)
        except Exception as exc:
            failure_count = int(context.metadata.get("sim_failure_count", "0") or 0) + 1
            context.metadata["sim_failure_count"] = str(failure_count)
            context.log(
                Message(
                    role=MessageRole.SYSTEM,
                    content=f"Simulation failed (attempt {failure_count}): {exc}. Using cached metrics when available.",
                    name="sim_failure_handler",
                )
            )
            if failure_count >= self._failure_limit:
                raise
            if self._last_successful_metrics:
                return dict(self._last_successful_metrics)
            return {}

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
                name="simulation_service",
            )
        )

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

    # Internal helpers

    def _register_tooling(self) -> None:
        self.tool_registry.register("prepare_ac_plan", self._tool_prepare_ac_plan)
        self.tool_registry.register("run_ngspice_simulation", self._tool_run_simulation)
        self.tool_registry.register("summarize_simulation_results", self._tool_summarize_results)

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
        logger.info(f"Running simulation tool. Deck: {deck.name if deck else 'default'}")
        raw_metrics = self.simulator.run(netlist_text, deck, sim_dir, artifact_dir=sim_dir)
        metrics = aggregate_measurement_values(raw_metrics)
        metrics = self._merge_with_cached_metrics(context, metrics)
        missing = self._warn_if_missing_metrics(context, metrics)
        state["raw_metrics"] = raw_metrics
        state["metrics"] = metrics
        if missing:
            state["missing_metrics"] = missing
        logger.info(f"Simulation tool finished. Metrics count: {len(metrics)}")
        return ToolResult(call_id=call.call_id or "run_ngspice", content=dict(metrics))

    def _tool_summarize_results(
        self, call: ToolCall, context: ExecutionContext, state: MutableMapping[str, Any]
    ) -> ToolResult:
        metrics = state.get("metrics", {})
        summary = describe_measurements(metrics if isinstance(metrics, Mapping) else {})
        state["simulation_summary"] = summary
        return ToolResult(call_id=call.call_id or "summary", content={"summary": summary})

    def _parse_param_changes(self, context: ExecutionContext) -> tuple[ParamChange, ...]:
        raw = context.metadata.get("param_changes")
        changes: list[ParamChange] = []
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                try:
                    changes.append(
                        ParamChange(
                            component=str(item.get("component")),
                            parameter=str(item.get("parameter")),
                            operation=str(item.get("operation", "set")),
                            value=float(item.get("value")),
                        )
                    )
                except Exception:
                    continue
        return tuple(changes)

    def _apply_structured_changes(self, netlist_text: str, context: ExecutionContext) -> str:
        changes = self._parse_param_changes(context)
        if not changes:
            return netlist_text
        try:
            patched, applied = apply_param_changes(netlist_text, changes)
            if applied:
                context.metadata["applied_param_changes"] = json.dumps(applied)
            return patched
        except Exception as exc:
            context.log(
                Message(
                    role=MessageRole.SYSTEM,
                    content=f"Structured netlist patch failed; falling back to original netlist. Error: {exc}",
                    name="netlist_patch",
                )
            )
            return netlist_text

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

    def _warn_if_missing_metrics(
        self, context: ExecutionContext, metrics: Mapping[str, float]
    ) -> Sequence[str]:
        missing = validate_metrics(metrics)
        if missing:
            context.log(
                Message(
                    role=MessageRole.SYSTEM,
                    content="Missing expected measurements: " + ", ".join(sorted(missing)),
                    name="metrics_validator",
                )
            )
        return missing

    def _ensure_netlist_loaded(self, context: ExecutionContext) -> str:
        if not context.netlist_path:
            raise ValueError("ExecutionContext.netlist_path must be set before running the agent")
        if not context.netlist_path.exists():
            raise FileNotFoundError(context.netlist_path)
        return context.netlist_path.read_text()

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


__all__ = ["SimulationService"]
