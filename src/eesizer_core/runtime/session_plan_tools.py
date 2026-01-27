from __future__ import annotations

from typing import Any, Mapping

from ..contracts.errors import ValidationError
from .tool_registry import ToolFn, ToolRegistry


def _unconfigured_tool(name: str) -> ToolFn:
    def _fn(_inputs: Mapping[str, Any], _ctx: Any, _params: Mapping[str, Any]) -> Mapping[str, Any]:
        raise ValidationError(f"tool '{name}' is not configured for execution")

    return _fn


# Step8: minimal deterministic tool whitelist for session planning/execution.
SESSION_TOOL_DEFS: dict[str, dict[str, Any]] = {
    "update_spec": {
        "description": "Apply a spec_delta (objectives only) to the current session spec (no phase run).",
        "params_schema": {"type": "object", "properties": {"spec_delta": {"type": "object"}}, "required": ["spec_delta"]},
        "io": {"inputs": ["session_run_dir", "source"], "outputs": ["session/spec_rev", "session/spec_hash"]},
        "constraints": ["objective-only changes", "no netlist/include changes"],
    },
    "update_cfg": {
        "description": "Apply a cfg_delta (budget + notes.grid_search/corner_validate only) to the current session cfg (no phase run).",
        "params_schema": {"type": "object", "properties": {"cfg_delta": {"type": "object"}}, "required": ["cfg_delta"]},
        "io": {"inputs": ["session_run_dir", "source"], "outputs": ["session/cfg_rev", "session/cfg_hash"]},
        "constraints": ["no netlist/include changes"],
    },
    "run_baseline": {
        "description": "Run/refresh phase p0_baseline using current session spec/cfg (checkpointed).",
        "params_schema": {"type": "object", "properties": {}},
        "io": {"inputs": ["session_run_dir", "source"], "outputs": ["p0/run_dir"]},
        "cost_model": {"iters": 1},
        "side_effects": ["writes a baseline run_dir + checkpoint", "updates session/meta_report.md"],
    },
    "run_grid_search": {
        "description": "Run/refresh phase p1_grid (grid search) using current session spec/cfg (checkpointed).",
        "params_schema": {"type": "object", "properties": {}},
        "io": {"inputs": ["session_run_dir", "source"], "outputs": ["p1/run_dir", "p1/topk", "p1/pareto"]},
        "cost_model": {"iters": "cfg.budget.max_iterations"},
        "side_effects": ["writes a grid run_dir + checkpoint", "updates session/meta_report.md"],
    },
    "run_corner_validate": {
        "description": "Run/refresh phase p2_corner_validate (validate topk/pareto candidates on corners) (checkpointed).",
        "params_schema": {"type": "object", "properties": {}},
        "io": {"inputs": ["session_run_dir", "source"], "outputs": ["p2/run_dir", "p2/robust_topk", "p2/robust_pareto"]},
        "cost_model": {"corners": "depends on cfg.notes.corner_validate"},
        "side_effects": ["writes robustness artifacts into the p1 run_dir", "updates session/meta_report.md"],
    },
}


def build_session_plan_registry(*, tool_fns: Mapping[str, ToolFn] | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    for name in sorted(SESSION_TOOL_DEFS.keys()):
        meta = SESSION_TOOL_DEFS[name]
        fn = tool_fns.get(name) if tool_fns and name in tool_fns else _unconfigured_tool(name)
        reg.register(
            name,
            fn,
            meta.get("params_schema"),
            description=str(meta.get("description") or ""),
            cost_model=dict(meta.get("cost_model") or {}),
            side_effects=list(meta.get("side_effects") or []),
            constraints=list(meta.get("constraints") or []),
            io=dict(meta.get("io") or {}),
        )
    return reg

