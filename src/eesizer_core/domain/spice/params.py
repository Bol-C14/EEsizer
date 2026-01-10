from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Pattern, Tuple
import re

from ...contracts.artifacts import CircuitIR, ParamDef, ParamSpace


def _compile_patterns(patterns: Iterable[str | Pattern[str]]) -> Tuple[Pattern[str], ...]:
    compiled: List[Pattern[str]] = []
    for p in patterns:
        compiled.append(re.compile(p) if isinstance(p, str) else p)
    return tuple(compiled)


class ParamInferenceRules:
    def __init__(
        self,
        *,
        allow_patterns: Iterable[str | Pattern[str]] = (),
        deny_patterns: Iterable[str | Pattern[str]] = (),
        default_units_by_key: Optional[Mapping[str, str]] = None,
        default_bounds_by_key: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
        wl_ratio_min: Optional[float] = None,
    ) -> None:
        self.allow_patterns = _compile_patterns(allow_patterns)
        self.deny_patterns = _compile_patterns(deny_patterns)
        self.default_units_by_key = dict(default_units_by_key or {})
        self.default_bounds_by_key = dict(default_bounds_by_key or {})
        self.wl_ratio_min = wl_ratio_min

    def allows(self, param_id: str) -> bool:
        if any(p.search(param_id) for p in self.deny_patterns):
            return False
        if not self.allow_patterns:
            return True
        return any(p.search(param_id) for p in self.allow_patterns)

    def unit_for(self, param_key: str, default: str) -> str:
        return self.default_units_by_key.get(param_key.lower(), default)

    def bounds_for(self, param_key: str) -> Tuple[Optional[float], Optional[float]]:
        return self.default_bounds_by_key.get(param_key.lower(), (None, None))


def infer_param_space_from_ir(
    cir: CircuitIR,
    *,
    default_unit: str = "",
    frozen_param_ids: Iterable[str] = (),
    rules: Optional[ParamInferenceRules] = None,
) -> ParamSpace:
    """Construct a ParamSpace from CircuitIR.param_locs with optional rules."""
    frozen = set(frozen_param_ids)
    rules = rules or ParamInferenceRules()
    param_defs: List[ParamDef] = []
    for param_id in sorted(cir.param_locs.keys()):
        if not rules.allows(param_id):
            continue
        key = param_id.split(".")[-1].lower()
        unit = rules.unit_for(key, default_unit)
        lower, upper = rules.bounds_for(key)
        param_defs.append(
            ParamDef(
                param_id=param_id,
                unit=unit,
                lower=lower,
                upper=upper,
                frozen=param_id in frozen,
                tags=(),
            )
        )
    return ParamSpace.build(param_defs)
