from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ...contracts.artifacts import CircuitIR, CircuitSpec, ParamDef, ParamSpace, Patch
from ...contracts.enums import PatchOpType
from ...contracts.errors import ValidationError
from ...contracts.guards import GuardCheck
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json
from ...domain.spice.patching import validate_patch, _parse_scalar_numeric


def _current_param_value(cir: CircuitIR, param_id: str) -> Optional[str]:
    loc = cir.param_locs.get(param_id.lower()) or cir.param_locs.get(param_id)
    if loc is None:
        return None
    raw = loc.raw_token
    start, end = loc.value_span
    if start < 0 or end > len(raw) or start > end:
        return None
    return raw[start:end]


def _requires_nonnegative(param_id: str, param_def: ParamDef | None) -> bool:
    if param_def is not None:
        if any(tag.lower() == "nonnegative" for tag in param_def.tags):
            return True
    parts = param_id.lower().split(".")
    if not parts:
        return False
    key = parts[-1]
    if key in ("w", "l"):
        return True
    if key == "value":
        base = parts[0]
        if base and base[0] in ("r", "c", "l"):
            return True
    return False


def _parse_numeric(value: object) -> float:
    parsed = _parse_scalar_numeric(value)  # uses shared domain parsing rules
    if not math.isfinite(parsed):
        raise ValidationError(f"non-finite numeric value '{value}'")
    return parsed


def _candidate_values_for_patch(
    cir: CircuitIR,
    patch: Patch,
    *,
    max_add_delta: float | None = None,
) -> tuple[dict[str, float], list[str]]:
    candidates: dict[str, float] = {}
    errors: list[str] = []
    for op in patch.ops:
        pid = op.param.lower()
        if op.op == PatchOpType.set:
            try:
                candidates[pid] = _parse_numeric(op.value)
            except ValidationError as exc:
                errors.append(f"param '{op.param}' has invalid set value '{op.value}': {exc}")
        elif op.op in (PatchOpType.add, PatchOpType.mul):
            current_text = _current_param_value(cir, pid)
            if current_text is None:
                errors.append(f"param '{op.param}' missing current value")
                continue
            try:
                current_val = _parse_numeric(current_text)
            except ValidationError as exc:
                errors.append(f"param '{op.param}' has non-numeric current value '{current_text}': {exc}")
                continue
            try:
                delta = _parse_numeric(op.value)
            except ValidationError as exc:
                errors.append(f"param '{op.param}' has invalid patch value '{op.value}': {exc}")
                continue
            if op.op == PatchOpType.add:
                if max_add_delta is not None and abs(delta) > max_add_delta:
                    errors.append(f"param '{op.param}' add delta {delta} exceeds max {max_add_delta}")
                candidates[pid] = current_val + delta
            else:
                candidates[pid] = current_val * delta
        else:
            errors.append(f"unsupported op '{op.op}' for param '{op.param}'")
    return candidates, errors


def _value_for_param(
    cir: CircuitIR,
    param_id: str,
    candidates: Mapping[str, float],
) -> tuple[Optional[float], Optional[str]]:
    pid = param_id.lower()
    if pid in candidates:
        return candidates[pid], None
    current_text = _current_param_value(cir, pid)
    if current_text is None:
        return None, f"param '{param_id}' missing current value"
    try:
        return _parse_numeric(current_text), None
    except ValidationError as exc:
        return None, f"param '{param_id}' has non-numeric current value '{current_text}': {exc}"


class PatchGuardOperator(Operator):
    """Pre-apply guard checks for patch validity and constraints."""

    name = "patch_guard"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        cir = inputs.get("circuit_ir")
        if not isinstance(cir, CircuitIR):
            raise ValidationError("PatchGuardOperator: 'circuit_ir' must be CircuitIR")
        param_space = inputs.get("param_space")
        if not isinstance(param_space, ParamSpace):
            raise ValidationError("PatchGuardOperator: 'param_space' must be ParamSpace")
        patch = inputs.get("patch")
        if not isinstance(patch, Patch):
            raise ValidationError("PatchGuardOperator: 'patch' must be Patch")

        spec = inputs.get("spec")
        if spec is not None and not isinstance(spec, CircuitSpec):
            raise ValidationError("PatchGuardOperator: 'spec' must be CircuitSpec when provided")

        guard_cfg = dict(inputs.get("guard_cfg") or {})
        max_patch_ops = guard_cfg.get("max_patch_ops", 20)
        max_mul_factor = guard_cfg.get("max_mul_factor", 10.0)
        max_add_delta = guard_cfg.get("max_add_delta")
        wl_ratio_min = guard_cfg.get("wl_ratio_min")

        errors: list[str] = []

        validation = validate_patch(
            cir,
            param_space,
            patch,
            wl_ratio_min=wl_ratio_min,
            max_mul_factor=max_mul_factor,
        )
        errors.extend(validation.errors)

        if max_patch_ops is not None and len(patch.ops) > max_patch_ops:
            errors.append(f"patch op count {len(patch.ops)} exceeds max {max_patch_ops}")

        candidates, parse_errors = _candidate_values_for_patch(cir, patch, max_add_delta=max_add_delta)
        errors.extend(parse_errors)

        for op in patch.ops:
            pid = op.param.lower()
            if pid not in candidates:
                continue
            param_def = param_space.get(pid)
            if _requires_nonnegative(pid, param_def) and candidates[pid] < 0:
                errors.append(f"param '{op.param}' must be non-negative")

        if spec is not None and spec.constraints:
            for constraint in spec.constraints:
                kind = constraint.kind
                data = dict(constraint.data)
                if kind == "param_range":
                    param_id = data.get("param")
                    if not isinstance(param_id, str):
                        errors.append("constraint param_range missing 'param'")
                        continue
                    lower = data.get("lower")
                    upper = data.get("upper")
                    value, seen_error = _value_for_param(cir, param_id, candidates)
                    if seen_error:
                        errors.append(seen_error)
                        continue
                    if value is None:
                        errors.append(f"constraint param_range missing value for '{param_id}'")
                        continue
                    if lower is not None:
                        try:
                            lower_val = float(lower)
                        except (TypeError, ValueError):
                            errors.append(f"constraint param_range invalid lower '{lower}'")
                        else:
                            if value < lower_val:
                                errors.append(f"constraint param_range '{param_id}' below lower {lower}")
                    if upper is not None:
                        try:
                            upper_val = float(upper)
                        except (TypeError, ValueError):
                            errors.append(f"constraint param_range invalid upper '{upper}'")
                        else:
                            if value > upper_val:
                                errors.append(f"constraint param_range '{param_id}' above upper {upper}")
                elif kind == "param_ratio_min":
                    lhs = data.get("lhs")
                    rhs = data.get("rhs")
                    min_ratio = data.get("min_ratio")
                    if not isinstance(lhs, str) or not isinstance(rhs, str):
                        errors.append("constraint param_ratio_min missing 'lhs'/'rhs'")
                        continue
                    if min_ratio is None:
                        errors.append("constraint param_ratio_min missing 'min_ratio'")
                        continue
                    try:
                        min_ratio_val = float(min_ratio)
                    except (TypeError, ValueError):
                        errors.append(f"constraint param_ratio_min invalid min_ratio '{min_ratio}'")
                        continue
                    lhs_val, lhs_err = _value_for_param(cir, lhs, candidates)
                    rhs_val, rhs_err = _value_for_param(cir, rhs, candidates)
                    if lhs_err or rhs_err:
                        if lhs_err:
                            errors.append(lhs_err)
                        if rhs_err:
                            errors.append(rhs_err)
                        continue
                    if rhs_val is None or lhs_val is None:
                        errors.append(f"constraint param_ratio_min missing values for '{lhs}'/'{rhs}'")
                        continue
                    if rhs_val == 0:
                        errors.append(f"constraint param_ratio_min divide by zero for '{rhs}'")
                        continue
                    if lhs_val < min_ratio_val * rhs_val:
                        errors.append(f"constraint param_ratio_min {lhs} < {min_ratio}*{rhs}")
                elif kind == "param_equal_group":
                    params = data.get("params")
                    tol = data.get("tol", 0.0)
                    if not isinstance(params, list) or not params:
                        errors.append("constraint param_equal_group missing 'params'")
                        continue
                    try:
                        tol_val = float(tol)
                    except (TypeError, ValueError):
                        errors.append(f"constraint param_equal_group invalid tol '{tol}'")
                        continue
                    values: list[float] = []
                    missing = False
                    for param_id in params:
                        if not isinstance(param_id, str):
                            errors.append("constraint param_equal_group param id is not a string")
                            missing = True
                            break
                        value, val_err = _value_for_param(cir, param_id, candidates)
                        if val_err:
                            errors.append(val_err)
                            missing = True
                            break
                        if value is None:
                            errors.append(f"constraint param_equal_group missing value for '{param_id}'")
                            missing = True
                            break
                        values.append(value)
                    if missing:
                        continue
                    if max(values) - min(values) > tol_val:
                        errors.append(f"constraint param_equal_group exceeds tol {tol}")
                else:
                    errors.append(f"constraint '{kind}' is not supported")

        ok = not errors
        check = GuardCheck(
            name="patch_guard",
            ok=ok,
            severity="hard",
            reasons=tuple(errors),
            data={"error_count": len(errors)},
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["param_space"] = ArtifactFingerprint(
            sha256=stable_hash_json([p.param_id for p in param_space.params])
        )
        provenance.inputs["patch"] = patch.fingerprint()
        provenance.inputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_json(sorted(cir.param_locs.keys()))
        )
        provenance.outputs["check"] = ArtifactFingerprint(
            sha256=stable_hash_json({"ok": check.ok, "severity": check.severity, "reasons": list(check.reasons)})
        )
        provenance.finish()

        return OperatorResult(outputs={"check": check}, provenance=provenance)
