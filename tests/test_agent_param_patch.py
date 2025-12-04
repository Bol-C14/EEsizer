import json
from pathlib import Path

from eesizer_core.agents.simple import OptimizationTargets, SimpleSizingAgent
from eesizer_core.config import AgentConfig, OptimizationConfig, SimulationConfig
from eesizer_core.context import ContextManager
from eesizer_core.simulation import MockNgSpiceSimulator


TEST_NETLIST = """
M1 out in 0 0 nch W=1u L=90n
R1 out 0 10k
"""


def build_agent(tmp_path: Path) -> SimpleSizingAgent:
    sim_cfg = SimulationConfig(binary_path=Path("ngspice"))
    opt_cfg = OptimizationConfig(max_iterations=2, tolerance_percent=0.05, vgs_margin_volts=0.05)
    agent_cfg = AgentConfig(
        name="test-agent",
        model="gpt-test",
        simulation=sim_cfg,
        optimization=opt_cfg,
        extra={"optimizer": {"failure_limit": 1}},
    )
    simulator = MockNgSpiceSimulator(sim_cfg)
    return SimpleSizingAgent(
        agent_cfg,
        simulator,
        goal="test param patch",
        targets=OptimizationTargets(gain_db=20.0, power_mw=1.0),
    )


def test_structured_param_changes_applied_before_sim(tmp_path: Path):
    netlist_path = tmp_path / "circuit.cir"
    netlist_path.write_text(TEST_NETLIST)
    agent = build_agent(tmp_path)
    with ContextManager(run_id="run", base_dir=tmp_path, config_name="test") as ctx:
        ctx.netlist_path = netlist_path
        ctx.metadata["param_changes"] = json.dumps(
            [
                {"component": "M1", "parameter": "W", "operation": "scale", "value": 1.5},
                {"component": "R1", "parameter": "R", "operation": "set", "value": 5000},
            ]
        )
        agent.run(ctx)
    # Original netlist remains unchanged
    original_text = netlist_path.read_text()
    assert "W=1u" in original_text and "10k" in original_text
    # Patched working copy is written under the run directory
    working_path = Path(ctx.metadata["working_netlist_path"])
    patched_text = working_path.read_text()
    assert "W=1.5" in patched_text or "W=1.5u" in patched_text
    assert ("5000" in patched_text) or ("5k" in patched_text)
