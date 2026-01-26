from .planning import group_metric_names_by_kind, merge_metrics, sim_plan_for_kind, extract_sim_plan
from .strategy import PatchLoopStrategy

# Backwards-compatible aliases for older imports.
_group_metric_names_by_kind = group_metric_names_by_kind
_merge_metrics = merge_metrics
_sim_plan_for_kind = sim_plan_for_kind
_extract_sim_plan = extract_sim_plan

__all__ = [
    "PatchLoopStrategy",
]
