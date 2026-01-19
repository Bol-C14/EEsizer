from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamDef, ParamSpace
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import GreedyCoordinatePolicy


def _make_obs(param_values: dict[str, float], score: float, param_space: ParamSpace, iteration: int, guard_report=None):
    src = CircuitSource(kind=SourceKind.spice_netlist, text="* test\n.end\n")
    spec = CircuitSpec()
    notes = {"param_values": dict(param_values), "current_score": score, "best_score": score}
    if guard_report is not None:
        notes["last_guard_report"] = guard_report
    return Observation(
        spec=spec,
        source=src,
        param_space=param_space,
        metrics=MetricsBundle(),
        iteration=iteration,
        notes=notes,
    )


def test_greedy_coordinate_policy_blacklists_frozen_param():
    ps = ParamSpace.build(
        [
            ParamDef(param_id="r1.value", lower=0.1, upper=10.0),
            ParamDef(param_id="c1.value", lower=0.1, upper=10.0),
        ]
    )
    policy = GreedyCoordinatePolicy(init_step=0.1)

    x = {"r1.value": 1.0, "c1.value": 1.0}
    score = 1.0
    obs1 = _make_obs(x, score, ps, 1)
    patch1 = policy.propose(obs1, ctx=None)
    assert patch1.stop is False
    assert patch1.ops[0].param == "r1.value"

    guard_report = {
        "ok": False,
        "checks": [
            {"name": "patch_guard", "ok": False, "reasons": ["param 'r1.value' is frozen"], "data": {}},
        ],
    }
    obs2 = _make_obs(x, score, ps, 2, guard_report=guard_report)
    patch2 = policy.propose(obs2, ctx=None)
    assert patch2.stop is False
    assert patch2.ops[0].param == "c1.value"
