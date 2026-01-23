from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .base import AgentContext


def _notes_block(notes: Mapping[str, Any], key: str) -> Dict[str, Any]:
    raw = notes.get(key)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


@dataclass
class SearchPlannerAgent:
    """Select which deterministic search strategy to run and with what config."""

    name: str = "search_planner"
    version: str = "0.1.0"

    def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        notes = ctx.cfg.notes or {}
        orch = notes.get("orchestrator")
        robust = False
        if isinstance(orch, Mapping):
            robust = bool(orch.get("robust", False))

        # Respect explicit user search configs when provided.
        user_grid = _notes_block(notes, "grid_search")
        user_corner = _notes_block(notes, "corner_search")

        selected_param_ids = list(inputs.get("selected_param_ids") or ())

        if robust:
            strategy = "corner_search"
            corner_cfg: Dict[str, Any] = {
                "mode": user_corner.get("mode", "coordinate"),
                "levels": int(user_corner.get("levels", 10)),
                "span_mul": float(user_corner.get("span_mul", 10.0)),
                "scale": user_corner.get("scale", "log"),
                "top_k": int(user_corner.get("top_k", 5)),
                "corners": user_corner.get("corners", "oat"),
            }
            if selected_param_ids:
                corner_cfg.setdefault("param_ids", selected_param_ids)
            return {
                "strategy": strategy,
                "cfg_notes": {"corner_search": corner_cfg},
                "mode": "robust" if robust else "default",
            }

        strategy = "grid_search"
        grid_cfg: Dict[str, Any] = {
            "mode": user_grid.get("mode", "coordinate"),
            "levels": int(user_grid.get("levels", 10)),
            "span_mul": float(user_grid.get("span_mul", 10.0)),
            "scale": user_grid.get("scale", "log"),
            "top_k": int(user_grid.get("top_k", 5)),
            "stop_on_first_pass": bool(user_grid.get("stop_on_first_pass", False)),
        }
        if selected_param_ids:
            grid_cfg.setdefault("param_ids", selected_param_ids)
        return {
            "strategy": strategy,
            "cfg_notes": {"grid_search": grid_cfg},
            "mode": "default",
        }