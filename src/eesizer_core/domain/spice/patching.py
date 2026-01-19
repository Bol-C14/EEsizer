from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re

from ...contracts.artifacts import CircuitIR, ParamSpace, Patch, PatchOp, Scalar, TokenLoc
from ...contracts.enums import PatchOpType
from ...contracts.errors import ValidationError
from .signature import topology_signature
from .tokenize import tokenize_spice_line
from .parse import index_spice_netlist


@dataclass
class PatchValidationResult:
    ok: bool
    errors: List[str]
    ratio_errors: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ratio_errors is None:
            self.ratio_errors = []


_UNIT_MULTIPLIERS = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}


def _current_param_value(cir: CircuitIR, param_id: str) -> Optional[str]:
    loc = cir.param_locs.get(param_id.lower())
    if loc is None:
        return None
    raw = loc.raw_token
    start, end = loc.value_span
    return raw[start:end]


def _candidate_or_current_numeric(
    cir: CircuitIR,
    candidate_values: Dict[str, float],
    param_id: str,
) -> tuple[Optional[float], Optional[str]]:
    if param_id in candidate_values:
        return candidate_values[param_id], None
    text = _current_param_value(cir, param_id)
    if text is None:
        return None, f"param '{param_id}' missing current value"
    try:
        return _parse_scalar_numeric(text), None
    except ValidationError:
        return None, f"param '{param_id}' has non-numeric current value '{text}'"


def validate_patch(
    cir: CircuitIR,
    param_space: ParamSpace,
    patch: Patch,
    *,
    wl_ratio_min: Optional[float] = None,
    max_mul_factor: Optional[float] = 10.0,
) -> PatchValidationResult:
    """
    检查 patch 是否：
    - 只引用已存在的 param_id
    - 不改 frozen 参数
    - 使用可支持的 op 类型
    - (预留) 不违反上下界 / ratio 约束
    """
    errors: List[str] = []
    candidate_values: Dict[str, float] = {}

    for op in patch.ops:
        pid = op.param.lower()
        if not param_space.contains(pid):
            errors.append(f"unknown param '{op.param}' (not in ParamSpace)")
            continue
        if pid not in cir.param_locs:
            errors.append(f"unknown param '{op.param}' (not in CircuitIR)")
            continue
        param_def = param_space.get(pid)
        if param_def is None:
            errors.append(f"unknown param '{op.param}' (ParamSpace lookup failed)")
            continue
        if param_def.frozen:
            errors.append(f"param '{op.param}' is frozen")
        if op.op not in (PatchOpType.set, PatchOpType.add, PatchOpType.mul):
            errors.append(f"unsupported op '{op.op}' for param '{op.param}'")
            continue

        current_val_text = _current_param_value(cir, pid)
        candidate_val: Optional[float] = None
        if op.op == PatchOpType.set:
            try:
                candidate_val = _parse_scalar_numeric(op.value)
            except ValidationError:
                errors.append(f"param '{op.param}' has non-numeric patch value '{op.value}'")
                candidate_val = None
        else:
            if current_val_text is None:
                errors.append(f"param '{op.param}' missing current value")
                continue
            try:
                current_val = _parse_scalar_numeric(current_val_text)
            except ValidationError:
                errors.append(f"param '{op.param}' has non-numeric current value '{current_val_text}'")
                continue
            try:
                delta = _parse_scalar_numeric(op.value)
            except ValidationError:
                errors.append(f"param '{op.param}' has non-numeric patch value '{op.value}'")
                continue
            if op.op == PatchOpType.add:
                candidate_val = current_val + delta
            elif op.op == PatchOpType.mul:
                candidate_val = current_val * delta
                if max_mul_factor is not None and abs(delta) > max_mul_factor:
                    errors.append(f"param '{op.param}' multiply factor {delta} exceeds max {max_mul_factor}")
                if delta <= 0:
                    errors.append(f"param '{op.param}' multiply factor must be positive")

        if candidate_val is not None:
            candidate_values[pid] = candidate_val
            if param_def.lower is not None and candidate_val < param_def.lower:
                errors.append(f"param '{op.param}' below lower bound {param_def.lower}")
            if param_def.upper is not None and candidate_val > param_def.upper:
                errors.append(f"param '{op.param}' above upper bound {param_def.upper}")

    ratio_errors: List[str] = []
    if wl_ratio_min is not None:
        for elem_name, loc in cir.param_locs.items():
            if elem_name.endswith(".w") or elem_name.endswith(".l"):
                parts = elem_name.split(".")
                if len(parts) < 2:
                    continue
                base = ".".join(parts[:-1])
                w_id = f"{base}.w"
                l_id = f"{base}.l"
                if w_id in cir.param_locs and l_id in cir.param_locs:
                    w_val, w_err = _candidate_or_current_numeric(cir, candidate_values, w_id)
                    if w_err:
                        errors.append(w_err)
                    l_val, l_err = _candidate_or_current_numeric(cir, candidate_values, l_id)
                    if l_err:
                        errors.append(l_err)
                    if w_val is not None and l_val is not None and w_val < wl_ratio_min * l_val:
                        ratio_errors.append(f"{w_id} < {wl_ratio_min}*{l_id}")

    all_errors = errors + ratio_errors
    return PatchValidationResult(ok=not all_errors, errors=all_errors, ratio_errors=ratio_errors)


def _parse_scalar_numeric(value: Scalar) -> float:
    if isinstance(value, bool):
        raise ValidationError("boolean values are not supported for numeric ops")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]+)?$", text)
    if not match:
        raise ValidationError(f"non-numeric value '{value}' for numeric op")
    number_str, suffix = match.groups()
    base = float(number_str)
    if not suffix:
        return base
    suffix_norm = suffix.lower()
    if suffix_norm in _UNIT_MULTIPLIERS:
        return base * _UNIT_MULTIPLIERS[suffix_norm]
    raise ValidationError(f"unsupported unit suffix '{suffix}' in value '{value}'")


def extract_param_values(
    cir: CircuitIR,
    param_ids: Optional[Iterable[str]] = None,
) -> tuple[dict[str, float], list[str]]:
    """Extract numeric param values from a CircuitIR."""
    values: dict[str, float] = {}
    errors: list[str] = []
    ids = list(param_ids) if param_ids is not None else list(cir.param_locs.keys())
    for param_id in ids:
        pid = param_id.lower()
        if pid not in cir.param_locs:
            continue
        raw = _current_param_value(cir, pid)
        if raw is None:
            continue
        try:
            values[pid] = _parse_scalar_numeric(raw)
        except ValidationError as exc:
            errors.append(f"param '{param_id}' has non-numeric value '{raw}': {exc}")
    return values, errors


def _format_scalar(value: Scalar) -> str:
    if isinstance(value, str):
        return value.strip()
    return repr(value)


def _token_span_at_index(line: str, token_idx: int) -> Tuple[int, int, str]:
    tokens, spans = tokenize_spice_line(line)
    if token_idx < 0 or token_idx >= len(tokens):
        raise ValidationError(f"token_idx {token_idx} out of range for line '{line.strip()}'")
    start, end = spans[token_idx]
    return start, end, tokens[token_idx]


def _apply_single_op_to_lines(
    cir: CircuitIR,
    lines: List[str],
    op: PatchOp,
) -> None:
    """在 lines 原地修改一个 PatchOp 对应的 token.

    约定：
    - 只用 cir.param_locs[op.param] 提供的 TokenLoc 来定位修改
    - 不改 key / node / element name，只改 value span
    """
    loc: TokenLoc | None = cir.param_locs.get(op.param)
    if loc is None:
        loc = cir.param_locs.get(op.param.lower())
    if loc is None:
        raise ValidationError(f"param '{op.param}' not found in CircuitIR")

    line = lines[loc.line_idx]
    token_start, token_end, token = _token_span_at_index(line, loc.token_idx)
    eq_pos = token.find("=")
    if eq_pos >= 0:
        start = eq_pos + 1
        end = len(token)
    else:
        start, end = loc.value_span
        if start < 0 or end > len(token) or start > end:
            raise ValidationError(f"invalid value_span for param '{op.param}'")
    old_val_text = token[start:end]
    if op.op == PatchOpType.set:
        new_val_text = _format_scalar(op.value)
    elif op.op == PatchOpType.add:
        old_val = _parse_scalar_numeric(old_val_text)
        delta = _parse_scalar_numeric(op.value)
        new_val_text = repr(old_val + delta)
    elif op.op == PatchOpType.mul:
        old_val = _parse_scalar_numeric(old_val_text)
        factor = _parse_scalar_numeric(op.value)
        new_val_text = repr(old_val * factor)
    else:
        raise ValidationError(f"unsupported op '{op.op}' for param '{op.param}'")

    new_token = f"{token[:start]}{new_val_text}{token[end:]}"
    lines[loc.line_idx] = f"{line[:token_start]}{new_token}{line[token_end:]}"


def apply_patch_to_ir(
    cir: CircuitIR,
    patch: Patch,
) -> CircuitIR:
    """
    在 CircuitIR 上应用 patch，生成一个新的 CircuitIR：

    - 复制一份 lines list
    - 依次对每个 op 调用 _apply_single_op_to_lines
    - 构造新 CircuitIR(lines', elements, param_locs, includes, warnings)
      （elements/param_locs 暂时可以复用旧的；如果未来支持新增参数，需要重新 index）
    """
    lines = list(cir.lines)
    for op in patch.ops:
        _apply_single_op_to_lines(cir, lines, op)

    netlist_text = "\n".join(lines)
    return index_spice_netlist(netlist_text, includes=cir.includes)


def apply_patch_with_topology_guard(
    cir: CircuitIR,
    patch: Patch,
    *,
    include_paths: bool = True,
    max_lines: int = 5000,
) -> CircuitIR:
    """应用 patch 并检查拓扑签名不变，否则抛 ValidationError."""
    old_sig_result = topology_signature("\n".join(cir.lines), include_paths=include_paths, max_lines=max_lines)
    new_cir = apply_patch_to_ir(cir, patch)
    new_sig_result = topology_signature("\n".join(new_cir.lines), include_paths=include_paths, max_lines=max_lines)
    if old_sig_result.signature != new_sig_result.signature:
        diff_parts = []
        old_elems = set(old_sig_result.circuit_ir.elements.keys())
        new_elems = set(new_sig_result.circuit_ir.elements.keys())
        if old_elems != new_elems:
            diff_parts.append(f"elements {old_elems ^ new_elems}")
        old_params = set(old_sig_result.circuit_ir.param_locs.keys())
        new_params = set(new_sig_result.circuit_ir.param_locs.keys())
        if old_params != new_params:
            diff_parts.append(f"params {old_params ^ new_params}")
        old_inc = set(old_sig_result.circuit_ir.includes)
        new_inc = set(new_sig_result.circuit_ir.includes)
        if old_inc != new_inc:
            diff_parts.append(f"includes {old_inc ^ new_inc}")
        reason = "; ".join(diff_parts) if diff_parts else "topology signature mismatch"
        raise ValidationError(f"Topology changed after patch: {reason}")
    return new_cir
