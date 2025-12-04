from pathlib import Path

from eesizer_core.agents.optimizer import MetricOptimizer
from eesizer_core.agents.scoring import OptimizationTargets, ScoringPolicy
from eesizer_core.config import AgentConfig, OptimizationConfig, SimulationConfig
from eesizer_core.context import ContextManager
from eesizer_core.prompts import PromptLibrary


class DummyPrompts(PromptLibrary):
    def __init__(self):
        super().__init__(search_paths=())

    def load(self, _name):
        class _Template:
            def render(self, **kwargs):
                return f"tpl {kwargs}"

        return _Template()


def _scoring():
    return ScoringPolicy(OptimizationTargets(gain_db=10.0, power_mw=1.0))


def test_optimizer_respects_pass_flag(tmp_path: Path):
    optimizer = MetricOptimizer(
        scoring=_scoring(),
        prompts=DummyPrompts(),
        targets=OptimizationTargets(gain_db=10.0, power_mw=1.0),
        max_iterations=5,
        nudge_fn=lambda m: dict(m),
    )
    with ContextManager("run", tmp_path, "test") as ctx:
        result = optimizer.optimize(ctx, {"gain_db": 12.0, "power_mw": 0.9, "pass_": True})
    assert result.history == ()
    assert result.metrics["gain_db"] == 12.0


def test_optimizer_stops_on_stagnation(tmp_path: Path):
    calls = {"count": 0}

    def nudge(metrics):
        calls["count"] += 1
        return dict(metrics)

    optimizer = MetricOptimizer(
        scoring=_scoring(),
        prompts=DummyPrompts(),
        targets=OptimizationTargets(gain_db=10.0, power_mw=1.0),
        max_iterations=10,
        nudge_fn=nudge,
        stagnation_rounds=2,
        min_improvement=0.05,
    )
    with ContextManager("run", tmp_path, "test") as ctx:
        result = optimizer.optimize(ctx, {"gain_db": 8.0, "power_mw": 1.0})
    # Should break after 2 stagnant rounds -> history length 2
    assert len(result.history) <= 2
    assert calls["count"] >= 1
