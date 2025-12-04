from pathlib import Path

from eesizer_core.agents.scoring import OptimizationTargets, ScoringPolicy
from eesizer_core.agents.simple import SimpleSizingAgent
from eesizer_core.config import AgentConfig, OptimizationConfig, SimulationConfig
from eesizer_core.simulation import MockNgSpiceSimulator


def _build_agent():
    sim_cfg = SimulationConfig(binary_path=Path("ngspice"))
    opt_cfg = OptimizationConfig(max_iterations=3, tolerance_percent=0.05, vgs_margin_volts=0.05)
    agent_cfg = AgentConfig(
        name="scoring-agent",
        model="gpt-test",
        simulation=sim_cfg,
        optimization=opt_cfg,
    )
    simulator = MockNgSpiceSimulator(sim_cfg)
    return SimpleSizingAgent(
        agent_cfg,
        simulator,
        goal="score test",
        targets=OptimizationTargets(gain_db=40.0, power_mw=5.0),
    )


def test_score_metrics_prefers_higher_gain_and_lower_power():
    agent = _build_agent()
    better = {"gain_db": 60.0, "power_mw": 2.5}
    worse = {"gain_db": 45.0, "power_mw": 8.0}
    assert agent.scoring.score(better) > agent.scoring.score(worse)


def test_scoring_policy_meets_targets_with_tolerance():
    policy = ScoringPolicy(OptimizationTargets(gain_db=50.0, power_mw=2.0), tolerance=0.1)
    assert policy.meets_targets({"gain_db": 46.0, "power_mw": 2.1})  # within tolerance
    assert not policy.meets_targets({"gain_db": 40.0, "power_mw": 2.5})
