from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import json

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str


def parse_json_strict(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc


def build_repair_prompt(*, schema: Mapping[str, Any], error: str, bad_output: str) -> str:
    schema_text = json.dumps(dict(schema), indent=2, sort_keys=True)
    return "\n".join(
        [
            "You are a JSON repair tool.",
            "",
            "Return ONLY a strict JSON object that matches this schema exactly:",
            schema_text,
            "",
            "The previous output was invalid for this reason:",
            error,
            "",
            "Previous output:",
            bad_output,
            "",
            "Return the repaired JSON now (no markdown, no comments).",
        ]
    )


def parse_and_validate(
    *,
    text: str,
    validate_fn: Callable[[Any], dict[str, Any]],
    schema_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        parsed_raw = parse_json_strict(text)
        parsed = validate_fn(parsed_raw)
    except Exception as exc:
        return None, {"ok": False, "schema": schema_name, "error": str(exc)}
    return parsed, {"ok": True, "schema": schema_name, "error": None}


@dataclass
class ParseJSONWithSchemaOperator(Operator):
    name: str = "parse_json_with_schema"
    version: str = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        text = inputs.get("text")
        if not isinstance(text, str):
            raise ValidationError("ParseJSONWithSchemaOperator requires 'text' string")
        schema_name = str(inputs.get("schema_name") or "schema")
        validate_fn = inputs.get("validate_fn")
        if not callable(validate_fn):
            raise ValidationError("ParseJSONWithSchemaOperator requires 'validate_fn' callable")

        parsed, validation = parse_and_validate(text=text, validate_fn=validate_fn, schema_name=schema_name)

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["text"] = ArtifactFingerprint(sha256=stable_hash_str(text))
        prov.inputs["schema_name"] = ArtifactFingerprint(sha256=stable_hash_json(schema_name))
        prov.outputs["validation"] = ArtifactFingerprint(sha256=stable_hash_json(validation))
        if parsed is not None:
            prov.outputs["parsed"] = ArtifactFingerprint(sha256=stable_hash_json(parsed))
        prov.finish()

        return OperatorResult(outputs={"parsed": parsed, "validation": validation}, provenance=prov)

