"""Parse and validate Patch JSON emitted by LLM policies."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from ..contracts import Patch, PatchOp
from ..contracts.enums import PatchOpType


@dataclass(frozen=True)
class PatchParseError(ValueError):
    """Raised when LLM patch JSON fails schema validation."""

    message: str

    def __str__(self) -> str:
        return self.message


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PatchParseError("empty LLM response")

    # Prefer fenced code blocks if present.
    for block in _CODE_FENCE_RE.findall(text):
        candidate = block.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        raise PatchParseError("patch JSON must be an object")

    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(payload, dict):
            return payload
        raise PatchParseError("patch JSON must be an object")

    raise PatchParseError("no JSON object found in LLM response")


def _expect_keys(obj: dict[str, Any], allowed: Iterable[str], required: Iterable[str] = ()) -> None:
    allowed_set = set(allowed)
    required_set = set(required)
    extra = set(obj.keys()) - allowed_set
    if extra:
        raise PatchParseError(f"unexpected keys: {sorted(extra)}")
    missing = required_set - set(obj.keys())
    if missing:
        raise PatchParseError(f"missing keys: {sorted(missing)}")


def parse_patch_json(text: str, allowed_params: set[str]) -> Patch:
    """Parse a Patch from LLM text, enforcing patch.schema.json constraints."""

    payload = _extract_json_object(text)
    _expect_keys(payload, allowed=("patch", "stop", "notes"), required=("patch",))

    patch_list = payload.get("patch")
    if not isinstance(patch_list, list):
        raise PatchParseError("patch must be a list")

    stop = payload.get("stop", False)
    if "stop" in payload and not isinstance(stop, bool):
        raise PatchParseError("stop must be a boolean if provided")

    notes = payload.get("notes", "")
    if "notes" in payload and not isinstance(notes, str):
        raise PatchParseError("notes must be a string if provided")

    ops: list[PatchOp] = []
    for idx, item in enumerate(patch_list):
        if not isinstance(item, dict):
            raise PatchParseError(f"patch[{idx}] must be an object")
        _expect_keys(item, allowed=("param", "op", "value", "why"), required=("param", "op", "value"))
        param = item.get("param")
        if not isinstance(param, str) or not param:
            raise PatchParseError(f"patch[{idx}].param must be a non-empty string")
        if param not in allowed_params:
            raise PatchParseError(f"patch[{idx}].param '{param}' is not in allowed param space")
        op = item.get("op")
        if not isinstance(op, str):
            raise PatchParseError(f"patch[{idx}].op must be a string")
        if op not in {t.value for t in PatchOpType}:
            raise PatchParseError(f"patch[{idx}].op must be one of {[t.value for t in PatchOpType]}")
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise PatchParseError(f"patch[{idx}].value must be number or string")
        why = item.get("why", "")
        if why is not None and not isinstance(why, str):
            raise PatchParseError(f"patch[{idx}].why must be a string if provided")
        ops.append(PatchOp(param=param, op=PatchOpType(op), value=value, why=why or ""))

    return Patch(ops=tuple(ops), stop=bool(stop), notes=notes or "")
