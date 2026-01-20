import pytest

from eesizer_core.policies.llm_patch_parser import PatchParseError, parse_patch_json


def test_parse_patch_json_basic():
    allowed = {"r1.value"}
    text = '{"patch":[{"param":"r1.value","op":"mul","value":0.9,"why":"tune"}],"notes":"ok"}'
    patch = parse_patch_json(text, allowed_params=allowed)
    assert patch.stop is False
    assert patch.notes == "ok"
    assert patch.ops[0].param == "r1.value"
    assert patch.ops[0].op.value == "mul"
    assert patch.ops[0].value == 0.9


def test_parse_patch_json_from_code_block():
    allowed = {"c1.value"}
    text = "```json\n{ \"patch\": [{\"param\":\"c1.value\",\"op\":\"set\",\"value\":\"1u\"}] }\n```"
    patch = parse_patch_json(text, allowed_params=allowed)
    assert patch.ops[0].param == "c1.value"
    assert patch.ops[0].op.value == "set"
    assert patch.ops[0].value == "1u"


def test_parse_patch_json_with_extra_text():
    allowed = {"r1.value"}
    text = "Here is the patch:\n{\"patch\": [{\"param\": \"r1.value\", \"op\": \"add\", \"value\": 1.5}]}\nThanks."
    patch = parse_patch_json(text, allowed_params=allowed)
    assert patch.ops[0].op.value == "add"


def test_parse_patch_json_rejects_unknown_param():
    allowed = {"r1.value"}
    text = '{"patch":[{"param":"c1.value","op":"mul","value":0.9}]}'
    with pytest.raises(PatchParseError):
        parse_patch_json(text, allowed_params=allowed)


def test_parse_patch_json_rejects_extra_fields():
    allowed = {"r1.value"}
    text = '{"patch":[{"param":"r1.value","op":"mul","value":0.9,"extra":true}]}'
    with pytest.raises(PatchParseError):
        parse_patch_json(text, allowed_params=allowed)


def test_parse_patch_json_rejects_invalid_op():
    allowed = {"r1.value"}
    text = '{"patch":[{"param":"r1.value","op":"scale","value":2.0}]}'
    with pytest.raises(PatchParseError):
        parse_patch_json(text, allowed_params=allowed)
