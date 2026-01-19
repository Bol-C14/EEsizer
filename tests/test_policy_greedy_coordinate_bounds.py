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


def test_greedy_coordinate_policy_clamps_to_bounds():
    ps = ParamSpace.build([ParamDef(param_id="r1.value", lower=1.0, upper=2.0)])
    policy = GreedyCoordinatePolicy(init_step=0.1)

    x = 1.99
    score = (x - 1.5) ** 2
    obs1 = _make_obs({"r1.value": x}, score, ps, 1)
    patch1 = policy.propose(obs1, ctx=None)
    op1 = patch1.ops[0]
    assert op1.op == PatchOpType.set
    assert float(op1.value) <= 2.0

    x = float(op1.value)
    score = (x - 1.5) ** 2
    obs2 = _make_obs({"r1.value": x}, score, ps, 2)
    patch2 = policy.propose(obs2, ctx=None)
    if not patch2.stop:
        op2 = patch2.ops[0]
        if op2.op == PatchOpType.set:
            assert float(op2.value) != x
