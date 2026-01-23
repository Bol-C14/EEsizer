from __future__ import annotations

from typing import Any, Mapping

from .types import CornerSearchConfig


def _as_id_list(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        return [str(v).lower() for v in value]
    return None


def parse_corner_search_config(notes: Mapping[str, Any]) -> CornerSearchConfig:
    raw = notes.get("corner_search")
    cfg = dict(raw) if isinstance(raw, Mapping) else {}

    mode = str(cfg.get("mode", "coordinate")).lower()
    levels = int(cfg.get("levels", 10))
    span_mul = float(cfg.get("span_mul", 10.0))
    scale = str(cfg.get("scale", "log")).lower()
    top_k = int(cfg.get("top_k", 5))
    stop_on_first_pass = bool(cfg.get("stop_on_first_pass", False))
    baseline_retries = int(cfg.get("baseline_retries", 0))
    corner_mode = str(cfg.get("corners", "oat")).lower()
    include_global_corners = bool(cfg.get("include_global_corners", False))
    override_mode = str(cfg.get("corner_override_mode", cfg.get("override_mode", "add"))).lower()
    require_baseline_corner_pass = bool(cfg.get("require_baseline_corner_pass", False))
    clamp_corner_overrides = bool(cfg.get("clamp_corner_overrides", True))

    search_param_ids = _as_id_list(cfg.get("search_param_ids"))
    if search_param_ids is None:
        search_param_ids = _as_id_list(cfg.get("param_ids"))
    corner_param_ids = _as_id_list(cfg.get("corner_param_ids"))

    return CornerSearchConfig(
        mode=mode,
        levels=levels,
        span_mul=span_mul,
        scale=scale,
        top_k=top_k,
        stop_on_first_pass=stop_on_first_pass,
        baseline_retries=baseline_retries,
        corner_mode=corner_mode,
        include_global_corners=include_global_corners,
        override_mode=override_mode,
        require_baseline_corner_pass=require_baseline_corner_pass,
        clamp_corner_overrides=clamp_corner_overrides,
        search_param_ids=search_param_ids,
        corner_param_ids=corner_param_ids,
    )
