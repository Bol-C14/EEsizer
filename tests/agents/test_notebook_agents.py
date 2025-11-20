import os
from pathlib import Path

import os
from pathlib import Path

import pytest

from eesizer_core.agents import Claude35Agent, Gemini30Agent, Gpt4oAgent, Gpt5Agent, Gpt5MiniAgent
from eesizer_core.agents.simple import OptimizationTargets
from eesizer_core.analysis.metrics import aggregate_measurement_values, validate_metrics
from eesizer_core.config import OptimizationConfig, OutputPathPolicy, SimulationConfig
from eesizer_core.context import ContextManager

AGENT_CLASSES = (Gpt5Agent, Gpt4oAgent, Gpt5MiniAgent, Claude35Agent, Gemini30Agent)


def _build_agent(agent_cls, simulator, output_root: Path):
    sim_cfg = SimulationConfig(binary_path=Path("ngspice"))
    opt_cfg = OptimizationConfig(max_iterations=2, tolerance_percent=0.1, vgs_margin_volts=0.05)
    policy = OutputPathPolicy(root=output_root)
    agent_cfg = agent_cls.default_config(sim_cfg, opt_cfg, output_paths=policy)
    return agent_cls(
        agent_cfg,
        simulator,
        goal="Increase gain while honoring power budget",
        targets=OptimizationTargets(gain_db=32.0, power_mw=3.0),
    )


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_agents_emit_notebook_prompts(agent_cls, mini_netlist, rich_mock_simulator, tmp_path):
    agent = _build_agent(agent_cls, rich_mock_simulator, tmp_path / "outputs")
    layout = agent.config.output_paths.build_layout("run1", netlist_stem=mini_netlist.stem)  # type: ignore[union-attr]
    with ContextManager("run1", tmp_path / "work", "test", path_layout=layout) as ctx:
        ctx.netlist_path = mini_netlist
        ctx.set_environment(corner="tt", supply_voltage=1.8, temperature_c=27.0)
        plan = agent.build_plan(ctx)
        assert any("gain" in message.content.lower() for message in plan)
        tools = agent.select_tools(ctx, plan)
        assert tools and "Tool-chain blueprint" in tools[0].content


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_agents_run_with_rich_metrics(agent_cls, mini_netlist, rich_mock_simulator, tmp_path):
    agent = _build_agent(agent_cls, rich_mock_simulator, tmp_path / "outputs")
    layout = agent.config.output_paths.build_layout("run2", netlist_stem=mini_netlist.stem)  # type: ignore[union-attr]
    with ContextManager("run2", tmp_path / "work", "test", path_layout=layout) as ctx:
        ctx.netlist_path = mini_netlist
        ctx.set_environment(corner="tt", supply_voltage=1.8, temperature_c=27.0)
        result = agent.run(ctx)

    assert result.success
    assert result.metrics is not None
    metrics = aggregate_measurement_values(result.metrics)
    assert metrics.get("gain_db", 0) >= 32.0
    assert pytest.approx(metrics.get("output_swing_v", 0), rel=0.01) == 1.0
    assert "cmrr_db" in metrics and "thd_db" in metrics
    assert not validate_metrics(metrics)


def test_recorded_responses_used_without_live_flag(recorded_llm_responses, recorded_only_env):
    assert recorded_only_env is None
    assert recorded_llm_responses["gpt5"]["plan"].startswith("recorded plan")
    use_live = bool(os.getenv("EESIZER_LIVE_LLM"))
    if use_live:
        pytest.skip("live LLM calls explicitly enabled")
    else:
        assert "recorded" in recorded_llm_responses["gemini30"]["tool_plan"]
