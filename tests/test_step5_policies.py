from eesizer_core.contracts import CircuitSource, CircuitSpec, ParamSpace, Patch, PatchOp, MetricsBundle
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import FixedSequencePolicy, RandomNudgePolicy


def _dummy_obs(param_space: ParamSpace | None = None) -> Observation:
    src = CircuitSource(kind=SourceKind.spice_netlist, text="* test\n.end\n")
    spec = CircuitSpec()
    ps = param_space or ParamSpace.build([])
    return Observation(spec=spec, source=src, param_space=ps, metrics=MetricsBundle(), iteration=1)


def test_fixed_sequence_policy_stops_after_sequence():
    patch1 = Patch(ops=(PatchOp(param="a", op=PatchOpType.set, value=1.0),))
    patch2 = Patch(ops=(PatchOp(param="b", op=PatchOpType.mul, value=2.0),))
    policy = FixedSequencePolicy(patches=[patch1, patch2])
    obs = _dummy_obs()

    p1 = policy.propose(obs, ctx=None)
    p2 = policy.propose(obs, ctx=None)
    p3 = policy.propose(obs, ctx=None)

    assert p1.ops[0].param == "a"
    assert p2.ops[0].param == "b"
    assert p3.stop is True


def test_random_nudge_policy_skips_frozen():
    from eesizer_core.contracts import ParamDef

    ps = ParamSpace.build(
        [
            ParamDef(param_id="m1.w", frozen=False),
            ParamDef(param_id="m1.l", frozen=True),
        ]
    )
    obs = _dummy_obs(param_space=ps)
    policy = RandomNudgePolicy(seed=123, step=0.05)
    patch = policy.propose(obs, ctx=None)

    assert patch.stop is False
    assert patch.ops[0].param == "m1.w"
    assert patch.ops[0].op == PatchOpType.mul
    assert patch.ops[0].value != 1.0

    empty_obs = _dummy_obs(param_space=ParamSpace.build([]))
    stop_patch = policy.propose(empty_obs, ctx=None)
    assert stop_patch.stop is True
