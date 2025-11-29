import json
import json
from pathlib import Path

from pathlib import Path

from eesizer_core.config import AgentConfig, OptimizationConfig, SimulationConfig
from eesizer_core.context import ArtifactKind, ContextManager
from eesizer_core.agents.simple import OptimizationTargets, SimpleSizingAgent
from eesizer_core.simulation import MockNgSpiceSimulator


TEST_NETLIST = """
* simple ota
M1 out in 0 0 nch W=1u L=90n
M2 out in vdd vdd pch W=2u L=90n
M3 out tail 0 0 nch W=0.5u L=90n
R1 vdd out 2k
"""


def build_agent(tmp_path: Path) -> SimpleSizingAgent:
    sim_cfg = SimulationConfig(binary_path=Path("ngspice"))
    opt_cfg = OptimizationConfig(max_iterations=5, tolerance_percent=0.05, vgs_margin_volts=0.05)
    agent_cfg = AgentConfig(
        name="test-agent",
        model="gpt-test",
        simulation=sim_cfg,
        optimization=opt_cfg,
    )
    simulator = MockNgSpiceSimulator(sim_cfg)
    return SimpleSizingAgent(
        agent_cfg,
        simulator,
        goal="Boost gain while keeping power low",
        targets=OptimizationTargets(gain_db=35.0, power_mw=3.0),
    )


def test_simple_agent_run(tmp_path):
    netlist_path = tmp_path / "ota.cir"
    netlist_path.write_text(TEST_NETLIST)
    agent = build_agent(tmp_path)

    with ContextManager(run_id="test", base_dir=tmp_path, config_name="test") as ctx:
        ctx.netlist_path = netlist_path
        ctx.set_environment(corner="tt", supply_voltage=1.8, temperature_c=27.0)
        result = agent.run(ctx)

    assert result.success
    assert result.metrics["gain_db"] >= 35.0
    assert result.artifacts["optimization_summary"].kind == ArtifactKind.OPTIMIZATION
    assert result.artifacts["netlist_copy"].kind == ArtifactKind.NETLIST
    assert Path(result.artifacts["netlist_copy"].path).exists()
    assert result.artifacts["simulation_summary"].kind == ArtifactKind.SIMULATION
    assert Path(result.artifacts["optimization_history_csv"].path).exists()


def test_simple_agent_uses_cached_metrics(tmp_path):
    netlist_path = tmp_path / "ota.cir"
    netlist_path.write_text(TEST_NETLIST)
    agent = build_agent(tmp_path)

    with ContextManager(run_id="test", base_dir=tmp_path, config_name="test") as ctx:
        ctx.netlist_path = netlist_path
        ctx.metadata["cached_metrics"] = json.dumps({"cmrr_db": 72.5})
        result = agent.run(ctx)

    assert result.success
    assert result.metrics["cmrr_db"] == 72.5
