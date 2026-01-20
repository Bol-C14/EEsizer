import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, Objective, ParamDef, ParamSpace
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import LLMPatchPolicy
from eesizer_core.policies.llm_patch_parser import PatchParseError


def _obs() -> Observation:
    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n")
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-1.0, sense="ge"),))
    param_space = ParamSpace.build([ParamDef(param_id="r1.value")])
    metrics = MetricsBundle()
    return Observation(
        spec=spec,
        source=source,
        param_space=param_space,
        metrics=metrics,
        iteration=1,
        notes={"current_score": 1.0, "param_values": {"r1.value": 1000.0}},
    )


def test_llm_patch_policy_parses_stop():
    policy = LLMPatchPolicy()
    patch = policy.parse_response('{"patch": [], "stop": true, "notes": "done"}', _obs())

    assert patch.stop is True
    assert patch.notes == "done"


def test_llm_patch_policy_rejects_unknown_param():
    policy = LLMPatchPolicy()
    with pytest.raises(PatchParseError, match="allowed param space"):
        policy.parse_response('{"patch":[{"param":"bad.value","op":"set","value":1}]}', _obs())


def test_llm_patch_policy_rejects_invalid_json():
    policy = LLMPatchPolicy()
    with pytest.raises(PatchParseError, match="no JSON object"):
        policy.parse_response("not json", _obs())
