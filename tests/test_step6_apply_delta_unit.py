from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.deltas import CfgDelta, SpecDelta
from eesizer_core.contracts.hashes import hash_cfg, hash_spec
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.apply_cfg_delta import apply_cfg_delta
from eesizer_core.operators.apply_spec_delta import apply_spec_delta


def test_apply_spec_delta_changes_hash_and_field() -> None:
    spec = CircuitSpec(objectives=(Objective(metric="phase_margin_deg", target=55.0, sense="ge"),))
    delta = SpecDelta.from_dict({"objectives": [{"metric": "phase_margin_deg", "op": "target", "value": 60.0}]})
    new_spec = apply_spec_delta(spec, delta)

    assert new_spec.objectives[0].target == 60.0
    assert hash_spec(new_spec) != hash_spec(spec)


def test_apply_cfg_delta_changes_hash_and_budget() -> None:
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=10), seed=0, notes={"grid_search": {"levels": 5}})
    delta = CfgDelta.from_dict({"budget": {"max_iterations": 20}, "notes": {"grid_search": {"levels": 7}}})
    new_cfg = apply_cfg_delta(cfg, delta)

    assert new_cfg.budget.max_iterations == 20
    assert new_cfg.notes["grid_search"]["levels"] == 7
    assert hash_cfg(new_cfg) != hash_cfg(cfg)

