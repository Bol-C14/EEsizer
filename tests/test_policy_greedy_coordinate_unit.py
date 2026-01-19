from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamDef, ParamSpace
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import GreedyCoordinatePolicy


def _make_obs(param_values: dict[str, float], score: float, param_space: ParamSpace, iteration: int) -> Observation:
    src = CircuitSource(kind=SourceKind.spice_netlist, text="* test\n.end\n")
    spec = CircuitSpec()
    notes = {"param_values": dict(param_values), "current_score": score, "best_score": score}
    return Observation(
        spec=spec,
        source=src,
        param_space=param_space,
        metrics=MetricsBundle(),
        iteration=iteration,
        notes=notes,
    )


def _apply_patch_value(value: float, op) -> float:
    if op.op == PatchOpType.set:
        return float(op.value)
    if op.op == PatchOpType.add:
        return value + float(op.value)
    if op.op == PatchOpType.mul:
        return value * float(op.value)
    raise AssertionError(f"unsupported op {op.op}")


def test_greedy_coordinate_policy_improves_score():
    ps = ParamSpace.build([ParamDef(param_id="r1.value", lower=0.1, upper=10.0)])
    policy = GreedyCoordinatePolicy(init_step=0.1, max_consecutive_success=3, max_trials_per_param=4)

    x = 1.0
    score = (x - 3.0) ** 2
    best_score = score
    factors: list[float] = []

    for i in range(1, 25):
        obs = _make_obs({"r1.value": x}, score, ps, i)
        patch = policy.propose(obs, ctx=None)
        assert patch.stop is False
        op = patch.ops[0]
        if op.op == PatchOpType.mul:
            factors.append(float(op.value))
        x = _apply_patch_value(x, op)
        score = (x - 3.0) ** 2
        best_score = min(best_score, score)

    assert best_score < 0.2
    assert any(f > 1.0 for f in factors)


def test_greedy_coordinate_policy_switches_direction_on_worse_score():
    ps = ParamSpace.build([ParamDef(param_id="r1.value", lower=0.1, upper=10.0)])
    policy = GreedyCoordinatePolicy(init_step=0.1)

    x = 5.0
    score = (x - 3.0) ** 2
    obs1 = _make_obs({"r1.value": x}, score, ps, 1)
    patch1 = policy.propose(obs1, ctx=None)
    op1 = patch1.ops[0]
    assert op1.op == PatchOpType.mul
    assert float(op1.value) > 1.0

    x = _apply_patch_value(x, op1)
    score = (x - 3.0) ** 2
    obs2 = _make_obs({"r1.value": x}, score, ps, 2)
    patch2 = policy.propose(obs2, ctx=None)
    op2 = patch2.ops[0]
    assert op2.op == PatchOpType.mul
    assert float(op2.value) < 1.0
