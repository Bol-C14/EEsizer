import json

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamDef, ParamSpace
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.policy import Observation
from eesizer_core.policies import LLMPatchPolicy
from eesizer_core.runtime.context import RunContext


def _make_obs(param_space: ParamSpace) -> Observation:
    src = CircuitSource(kind=SourceKind.spice_netlist, text="* test\n.end\n")
    spec = CircuitSpec()
    metrics = MetricsBundle()
    notes = {"param_values": {"r1.value": 1000.0}, "current_score": 1.0, "best_score": 1.0, "attempt": 0}
    return Observation(spec=spec, source=src, param_space=param_space, metrics=metrics, iteration=1, notes=notes)


def test_llm_patch_policy_retries_on_parse_error(tmp_path):
    param_space = ParamSpace.build([ParamDef(param_id="r1.value")])
    obs = _make_obs(param_space)
    ctx = RunContext(workspace_root=tmp_path)

    policy = LLMPatchPolicy(
        provider="mock",
        mock_responses=[
            "not json",
            '{"patch":[{"param":"r1.value","op":"mul","value":0.9}]}',
        ],
        max_retries=1,
    )

    patch = policy.propose(obs, ctx=ctx)
    assert patch.stop is False
    assert patch.ops[0].param == "r1.value"

    stage_dir = ctx.run_dir() / "llm" / "llm_i001_a00"
    retry_dir = ctx.run_dir() / "llm" / "llm_i001_a00_r01"
    assert (stage_dir / "parse_error.txt").exists()
    assert (retry_dir / "parsed_patch.json").exists()

    calls_path = ctx.run_dir() / "provenance" / "operator_calls.jsonl"
    lines = calls_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    llm_calls = [p for p in payloads if p.get("operator") == "llm_call"]
    assert len(llm_calls) >= 2
