from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re

from ...contracts.artifacts import (
    CircuitIR,
    ParamSpace,
    Patch,
    PatchOp,
    PatchOpType,
    Scalar,
    TokenLoc,
)
from ...contracts.errors import ValidationError
from .signature import topology_signature


@dataclass
class PatchValidationResult:
    ok: bool
    errors: List[str]


def validate_patch(
    cir: CircuitIR,
    param_space: ParamSpace,
    patch: Patch,
) -> PatchValidationResult:
    """
    检查 patch 是否：
    - 只引用已存在的 param_id
    - 不改 frozen 参数
    - 使用可支持的 op 类型
    - (预留) 不违反上下界 / ratio 约束
    """
    errors: List[str] = []

    for op in patch.ops:
        if not param_space.contains(op.param):
            errors.append(f"unknown param '{op.param}' (not in ParamSpace)")
            continue
        if op.param not in cir.param_locs:
            errors.append(f"unknown param '{op.param}' (not in CircuitIR)")
            continue
        param_def = param_space.get(op.param)
        if param_def is None:
            errors.append(f"unknown param '{op.param}' (ParamSpace lookup failed)")
            continue
        if param_def.frozen:
            errors.append(f"param '{op.param}' is frozen")
        if op.op not in (PatchOpType.set, PatchOpType.add, PatchOpType.mul):
            errors.append(f"unsupported op '{op.op}' for param '{op.param}'")

        # TODO: check bounds/ratio when ParamDef constraints are active.

    return PatchValidationResult(ok=not errors, errors=errors)


def _parse_scalar_numeric(value: Scalar) -> float:
    if isinstance(value, bool):
        raise ValidationError("boolean values are not supported for numeric ops")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValidationError(f"non-numeric value '{value}' for numeric op") from exc


def _format_scalar(value: Scalar) -> str:
    if isinstance(value, str):
        return value.strip()
    return repr(value)


def _token_span_at_index(line: str, token_idx: int) -> Tuple[int, int, str]:
    matches = list(re.finditer(r"\S+", line))
    if token_idx < 0 or token_idx >= len(matches):
        raise ValidationError(f"token_idx {token_idx} out of range for line '{line.strip()}'")
    match = matches[token_idx]
    return match.start(), match.end(), match.group(0)


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
        raise ValidationError(f"param '{op.param}' not found in CircuitIR")

    line = lines[loc.line_idx]
    token_start, token_end, token = _token_span_at_index(line, loc.token_idx)
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

    return CircuitIR(
        lines=tuple(lines),
        elements=cir.elements,
        param_locs=cir.param_locs,
        includes=cir.includes,
        warnings=cir.warnings,
    )


def apply_patch_with_topology_guard(
    cir: CircuitIR,
    patch: Patch,
    *,
    include_paths: bool = True,
) -> CircuitIR:
    """应用 patch 并检查拓扑签名不变，否则抛 ValidationError."""
    old_sig = topology_signature("\n".join(cir.lines), include_paths=include_paths).signature
    new_cir = apply_patch_to_ir(cir, patch)
    new_sig = topology_signature("\n".join(new_cir.lines), include_paths=include_paths).signature
    if old_sig != new_sig:
        raise ValidationError("Topology changed after patch")
    return new_cir
