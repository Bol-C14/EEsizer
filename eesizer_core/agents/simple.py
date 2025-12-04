"""Reference implementation of the Agent lifecycle using a pluggable simulator."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, MutableMapping, Sequence

from ..config import AgentConfig, ToolConfig
from ..context import ExecutionContext
from ..messaging import Message, MessageRole, ToolCall
from ..prompts import PromptLibrary
from ..providers import LLMProvider, LLMResponse, build_provider
from ..simulation import MockNgSpiceSimulator, NgSpiceRunner
from .base import Agent, AgentMetadata
from .optimizer import MetricOptimizer
from .scoring import OptimizationTargets, ScoringPolicy
from .services import (
    OptimizationService,
    PlannerService,
    SimulationService,
    ToolSelectionService,
)
import logging


logger = logging.getLogger(__name__)


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
        self._summary = None
        self.prompts = PromptLibrary(
            search_paths=self.config.prompt_paths,
            overrides=self.config.prompt_overrides,
        )
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
        self._failure_limit = self._optimizer_param("failure_limit", default=2, cast=int)
        self.planner_service = PlannerService(self.prompts, self._chat_with_retries)
        self.tool_selector_service = ToolSelectionService(self.prompts, self._chat_with_retries)
        self.simulation_service = SimulationService(
            simulator,
            self.prompts,
            default_tool_calls=self._default_tool_calls(),
            tool_schemas=self._tool_schemas(),
            failure_limit=self._failure_limit,
        )
        self.optimization_service = OptimizationService(
            optimizer=self.optimizer,
            scoring=self.scoring,
            prompts=self.prompts,
            targets=self.targets,
            goal=self.goal,
            tolerance_percent=self.config.optimization.tolerance_percent,
        )

    def build_plan(self, context: ExecutionContext) -> Sequence[Message]:
        """Emit a planning note that assumes ``context.netlist_path`` is set."""

        netlist_text = self.simulation_service._ensure_netlist_loaded(context)
        messages, summary = self.planner_service.build_plan(
            context,
            context.netlist_path,  # type: ignore[arg-type]
            netlist_text,
            self.goal,
            self.targets,
        )
        self._summary = summary
        return list(messages)

    def select_tools(self, context: ExecutionContext, history: Sequence[Message]) -> Sequence[Message]:
        """Describe the simulator/tooling choices used by downstream stages."""

        summary_desc = self._summary.describe() if self._summary else ""
        messages = self.tool_selector_service.select_tools(
            context,
            agent_name=self.metadata.name,
            netlist_summary=summary_desc,
            default_tool_plan=self._default_tool_plan(),
            tool_schemas=self._tool_schemas(),
        )
        return list(messages)

    def run_simulation(self, context: ExecutionContext) -> MutableMapping[str, float]:
        """Run the deterministic simulator and log telemetry in the execution context."""

        return self.simulation_service.run(context)

    def optimize(self, context: ExecutionContext, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        """Iteratively nudge metrics upward while mirroring the notebook log style."""

        return self.optimization_service.optimize(context, metrics)

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
    def _default_tool_plan() -> list[Mapping[str, object]]:
        return [
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

    def _chat_with_retries(
        self,
        messages: Sequence[Message],
        *,
        response_name: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> LLMResponse:
        attempts = 0
        cfg = getattr(self.config, "extra", {}) or {}
        max_attempts = cfg.get("llm_retries", 2) or 1
        backoff = float(cfg.get("llm_retry_backoff", 1.0) or 1.0)
        formatting_guard = Message(
            role=MessageRole.SYSTEM,
            content="Return strictly formatted output (JSON/tool calls only). Avoid prose, explanations, or markdown fences.",
            name="format_guard",
        )
        last_error: Exception | None = None
        while attempts < max_attempts:
            attempts += 1
            try:
                payload = list(messages)
                if attempts > 1:
                    payload = [formatting_guard] + payload
                response = self.llm.chat(payload, response_name=response_name, tools=tools)
                missing_tools = bool(tools) and not (response.message.tool_calls or ())
                missing_content = not response.message.content and not missing_tools
                if (missing_tools or missing_content) and attempts < max_attempts:
                    time.sleep(backoff)
                    continue
                return response
            except Exception as exc:  # pragma: no cover - live LLM failures
                last_error = exc
                if attempts < max_attempts:
                    time.sleep(backoff)
                    continue
        if last_error:
            raise last_error
        return self.llm.chat(messages, response_name=response_name, tools=tools)


__all__ = ["SimpleSizingAgent", "OptimizationTargets", "ScoringPolicy"]
