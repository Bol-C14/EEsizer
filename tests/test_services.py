import json
from pathlib import Path

from eesizer_core.agents.services import (
    OptimizationService,
    PlannerService,
    SimulationService,
    ToolSelectionService,
)
from eesizer_core.agents.optimizer import MetricOptimizer
from eesizer_core.agents.scoring import OptimizationTargets, ScoringPolicy
from eesizer_core.agents.simple import SimpleSizingAgent
from eesizer_core.config import SimulationConfig
from eesizer_core.context import ContextManager, ExecutionContext
from eesizer_core.messaging import Message, MessageRole, ToolCall
from eesizer_core.prompts import PromptLibrary
from eesizer_core.providers import LLMResponse
from eesizer_core.simulation import MockNgSpiceSimulator


class StubChat:
    def __init__(self, content: str = "ok"):
        self.calls = []
        self.content = content
        self.tool_calls: tuple[ToolCall, ...] = ()

    def __call__(self, messages, *, response_name=None, tools=None):
        self.calls.append(
            {"messages": list(messages), "response_name": response_name, "tools": tools}
        )
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=self.content, tool_calls=self.tool_calls)
        )


def test_planner_service_builds_plan(tmp_path: Path):
    netlist_text = "M1 out in 0 0 nch W=1u L=90n\nR1 out 0 1k\n"
    netlist_path = tmp_path / "circuit.cir"
    netlist_path.write_text(netlist_text)
    ctx = ExecutionContext(run_id="run", working_dir=tmp_path, config_name="test")

    chat = StubChat(content="plan-response")
    service = PlannerService(PromptLibrary(), chat)
    messages, summary = service.build_plan(
        ctx,
        netlist_path,
        netlist_text,
        goal="maximize gain",
        targets=OptimizationTargets(gain_db=30.0, power_mw=5.0),
    )

    assert summary.total_components == 2
    assert len(messages) == 3
    assert chat.calls[0]["response_name"] == "plan"
    assert ctx.messages.messages == []  # service leaves history untouched


def test_tool_selection_service_uses_schema_and_plan(tmp_path: Path):
    ctx = ExecutionContext(run_id="run", working_dir=tmp_path, config_name="test")
    chat = StubChat(content="tool-plan")
    default_plan = [{"name": "prepare_ac_plan", "arguments": {"deck_name": "deck"}}]
    tool_schemas = ({"name": "prepare_ac_plan", "parameters": {}},)

    service = ToolSelectionService(PromptLibrary(), chat)
    messages = service.select_tools(
        ctx,
        agent_name="agent",
        netlist_summary="Netlist summary text",
        default_tool_plan=default_plan,
        tool_schemas=tool_schemas,
    )

    assert len(messages) == 2
    assert '"prepare_ac_plan"' in messages[0].content
    assert chat.calls[0]["tools"] == tool_schemas
    assert ctx.messages.messages == []  # no implicit logging


def test_simulation_service_runs_and_attaches_artifacts(tmp_path: Path):
    netlist_text = "M1 out in 0 0 nch W=1u L=90n\nR1 out 0 1k\n"
    netlist_path = tmp_path / "circuit.cir"
    netlist_path.write_text(netlist_text)
    sim_cfg = SimulationConfig(binary_path=Path("ngspice"))
    simulator = MockNgSpiceSimulator(sim_cfg)
    default_calls = SimpleSizingAgent._default_tool_calls()
    schemas = SimpleSizingAgent._tool_schemas()

    with ContextManager(run_id="run", base_dir=tmp_path, config_name="test") as ctx:
        ctx.netlist_path = netlist_path
        service = SimulationService(
            simulator,
            PromptLibrary(),
            default_tool_calls=default_calls,
            tool_schemas=schemas,
        )
        metrics = service.run(ctx)

    assert metrics["gain_db"] > 0
    assert "simulation_metrics" in ctx.artifacts
    assert Path(ctx.artifacts["simulation_metrics"].path).exists()
    assert "simulation_summary" in ctx.artifacts
    assert ctx.metadata.get("sim_output")


def test_optimization_service_generates_reports(tmp_path: Path):
    targets = OptimizationTargets(gain_db=20.0, power_mw=5.0)
    scoring = ScoringPolicy(targets=targets, tolerance=0.05)
    prompts = PromptLibrary()
    optimizer = MetricOptimizer(
        scoring=scoring,
        prompts=prompts,
        targets=targets,
        max_iterations=2,
        nudge_fn=lambda m: {
            "gain_db": m.get("gain_db", 0) + 5,
            "power_mw": max(0.1, m.get("power_mw", 5) - 0.5),
        },
    )
    service = OptimizationService(
        optimizer=optimizer,
        scoring=scoring,
        prompts=prompts,
        targets=targets,
        goal="improve gain",
        tolerance_percent=0.05,
    )

    netlist_path = tmp_path / "circuit.cir"
    netlist_path.write_text("M1 out in 0 0 nch W=1u L=90n\n")
    with ContextManager(run_id="run", base_dir=tmp_path, config_name="test") as ctx:
        ctx.netlist_path = netlist_path
        optimized = service.optimize(ctx, {"gain_db": 5.0, "power_mw": 5.0})

        # Gains are nudged but not forced to targets; flags capture status.
        assert optimized["gain_db"] < targets.gain_db
        assert optimized["targets_met"] == 0.0
        assert "optimization_summary" in ctx.artifacts
        assert Path(ctx.artifacts["optimization_summary"].path).exists()
        assert "netlist_copy" in ctx.artifacts
        assert Path(ctx.artifacts["netlist_copy"].path).exists()
