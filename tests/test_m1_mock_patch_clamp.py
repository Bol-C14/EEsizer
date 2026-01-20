from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamDef, ParamSpace
from eesizer_core.contracts.enums import PatchOpType, SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import LLMPatchPolicy


def _make_obs(param_space: ParamSpace, current_value: float, current_score: float) -> Observation:
    src = CircuitSource(kind=SourceKind.spice_netlist, text="* test\n.end\n")
    spec = CircuitSpec()
    metrics = MetricsBundle()
    notes = {
        "param_values": {"r1.value": current_value},
        "current_score": current_score,
        "best_score": current_score,
        "attempt": 0,
    }
    return Observation(spec=spec, source=src, param_space=param_space, metrics=metrics, iteration=1, notes=notes)


def test_mock_patch_clamps_to_lower_bound():
    param_space = ParamSpace.build([ParamDef(param_id="r1.value", lower=1e-6, upper=1e3)])
    obs = _make_obs(param_space, current_value=1e-7, current_score=1.0)
    policy = LLMPatchPolicy(provider="mock", max_retries=0)

    request, stop_reason = policy.build_request(obs)
    assert stop_reason is None
    assert request is not None
    prompt = request.get("user", "")
    response_text = policy.mock_response(prompt, obs)
    patch = policy.parse_response(response_text, obs)
    assert patch.stop is False
    assert patch.ops[0].op == PatchOpType.set
    assert patch.ops[0].value == 1e-6
