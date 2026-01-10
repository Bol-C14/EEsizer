from __future__ import annotations

from typing import Iterable, List

from ...contracts.artifacts import CircuitIR, ParamDef, ParamSpace


def infer_param_space_from_ir(
    cir: CircuitIR,
    *,
    default_unit: str = "",
    frozen_param_ids: Iterable[str] = (),
) -> ParamSpace:
    """Construct a minimal ParamSpace from CircuitIR.param_locs.

    - Each param_id in CircuitIR.param_locs becomes a ParamDef entry.
    - Unit uses default_unit until device-aware logic is added.
    - Bounds are left unset (None) for later constraint injection.
    - frozen flags are set from frozen_param_ids.
    """
    frozen = set(frozen_param_ids)
    param_defs: List[ParamDef] = []
    for param_id in sorted(cir.param_locs.keys()):
        param_defs.append(
            ParamDef(
                param_id=param_id,
                unit=default_unit,
                lower=None,
                upper=None,
                frozen=param_id in frozen,
                tags=(),
            )
        )
    return ParamSpace.build(param_defs)
