from __future__ import annotations

from typing import Any, Mapping

from ..contracts.errors import ValidationError
from ..contracts.grid_search_config import GridSearchConfig


def _coerce_int(value: Any, name: str, *, min_value: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an int") from exc
    if out < min_value:
        raise ValidationError(f"{name} must be >= {min_value}")
    return out


def _coerce_float(value: Any, name: str, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a float") from exc
    if min_value is not None and out < min_value:
        raise ValidationError(f"{name} must be >= {min_value}")
    return out


def _normalize_str(value: Any, name: str, *, allowed: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    out = value.strip().lower()
    if allowed is not None and out not in allowed:
        raise ValidationError(f"{name} must be one of {', '.join(allowed)}")
    return out


def _normalize_param_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{name} must be a list/tuple of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(f"{name} entries must be non-empty strings")
        pid = item.strip().lower()
        if pid not in out:
            out.append(pid)
    return tuple(out)


def _normalize_numeric_by_param(value: Any, name: str) -> tuple[float, dict[str, float]]:
    if isinstance(value, Mapping):
        per_param: dict[str, float] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            per_param[key.lower()] = _coerce_float(val, f"{name}[{key}]")
        return 0.0, per_param
    return _coerce_float(value, name), {}


def _normalize_int_by_param(value: Any, name: str) -> tuple[int, dict[str, int]]:
    if isinstance(value, Mapping):
        per_param: dict[str, int] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            per_param[key.lower()] = _coerce_int(val, f"{name}[{key}]")
        return 0, per_param
    return _coerce_int(value, name), {}


def _normalize_scale_by_param(value: Any, name: str) -> tuple[str, dict[str, str]]:
    if isinstance(value, Mapping):
        per_param: dict[str, str] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            per_param[key.lower()] = _normalize_str(val, f"{name}[{key}]", allowed=("log", "linear"))
        return "log", per_param
    return _normalize_str(value, name, allowed=("log", "linear")), {}


def parse_grid_search_config(notes: Mapping[str, Any], seed: int | None) -> GridSearchConfig:
    raw = notes.get("grid_search")
    if not isinstance(raw, Mapping):
        raw = {}

    mode = _normalize_str(raw.get("mode", "coordinate"), "mode", allowed=("coordinate", "factorial"))
    levels, per_param_levels = _normalize_int_by_param(raw.get("levels", 10), "levels")
    span_mul, per_param_span_mul = _normalize_numeric_by_param(raw.get("span_mul", 10.0), "span_mul")
    scale, per_param_scale = _normalize_scale_by_param(raw.get("scale", "log"), "scale")
    top_k = _coerce_int(raw.get("top_k", 5), "top_k")
    stop_on_first_pass = bool(raw.get("stop_on_first_pass", False))
    baseline_retries = _coerce_int(raw.get("baseline_retries", 0), "baseline_retries", min_value=0)
    continue_after_baseline_pass = bool(raw.get("continue_after_baseline_pass", False))
    max_params = _coerce_int(raw.get("max_params", 8), "max_params")

    max_candidates_raw = raw.get("max_candidates")
    max_candidates = None if max_candidates_raw is None else _coerce_int(max_candidates_raw, "max_candidates")

    param_ids = _normalize_param_list(raw.get("param_ids"), "param_ids")
    recommended_knobs = _normalize_param_list(raw.get("recommended_knobs"), "recommended_knobs")
    param_select_policy = _normalize_str(
        raw.get("param_select_policy", "recommended"),
        "param_select_policy",
        allowed=("recommended", "explicit", "auto_truncate"),
    )
    truncate_policy = _normalize_str(
        raw.get("truncate_policy", "seed_shuffle"),
        "truncate_policy",
        allowed=("seed_shuffle", "lexicographic"),
    )
    include_nominal = bool(raw.get("include_nominal", False))
    allow_param_ids_override_frozen = bool(raw.get("allow_param_ids_override_frozen", False))

    seed_value = raw.get("seed", seed)
    if seed_value is None:
        seed_value = 0
    seed_final = _coerce_int(seed_value, "seed", min_value=0)

    return GridSearchConfig(
        mode=mode,
        levels=levels,
        span_mul=span_mul,
        scale=scale,
        top_k=top_k,
        stop_on_first_pass=stop_on_first_pass,
        baseline_retries=baseline_retries,
        continue_after_baseline_pass=continue_after_baseline_pass,
        max_params=max_params,
        max_candidates=max_candidates,
        param_ids=param_ids,
        recommended_knobs=recommended_knobs,
        param_select_policy=param_select_policy,
        truncate_policy=truncate_policy,
        include_nominal=include_nominal,
        allow_param_ids_override_frozen=allow_param_ids_override_frozen,
        seed=seed_final,
        per_param_levels=per_param_levels,
        per_param_span_mul=per_param_span_mul,
        per_param_scale=per_param_scale,
    )
