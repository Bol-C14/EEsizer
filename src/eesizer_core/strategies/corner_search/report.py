from __future__ import annotations

from typing import Any, Mapping

from ...analysis.pareto import top_k
from ...contracts import MetricsBundle
from ...metrics.reporting import format_metric_line, metric_definition_lines
from ...contracts.strategy import StrategyConfig


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "-"
    parts = [f"{name}={value}" for name, value in metrics.items()]
    return ", ".join(parts)


def build_corner_report(
    *,
    cfg: StrategyConfig,
    corner_cfg: Mapping[str, Any],
    baseline_eval: dict[str, Any],
    baseline_metrics: MetricsBundle,
    baseline_summary: Mapping[str, Any],
    candidate_entries: list[dict[str, Any]],
    pareto_entries: list[dict[str, Any]],
    sim_runs_failed: int,
    param_value_errors: list[str],
) -> list[str]:
    lines: list[str] = []
    lines.append("# Corner Search Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- max_iterations: {cfg.budget.max_iterations}")
    lines.append(f"- mode: {corner_cfg.get('mode', 'coordinate')}")
    lines.append(f"- levels: {corner_cfg.get('levels', 10)}")
    lines.append(f"- span_mul: {corner_cfg.get('span_mul', 10.0)}")
    lines.append(f"- scale: {corner_cfg.get('scale', 'log')}")
    lines.append(f"- corners: {corner_cfg.get('corners', 'oat')}")
    lines.append(f"- include_global_corners: {corner_cfg.get('include_global_corners', False)}")
    lines.append(f"- override_mode: {corner_cfg.get('corner_override_mode', corner_cfg.get('override_mode', 'add'))}")
    lines.append("")

    definition_lines = metric_definition_lines(baseline_metrics.values.keys())
    if definition_lines:
        lines.extend(definition_lines)
        lines.append("")

    lines.append("## Baseline (Nominal)")
    lines.append(f"- score: {baseline_eval.get('score')}")
    lines.append(f"- all_pass: {baseline_eval.get('all_pass')}")
    lines.append(f"- worst_score: {baseline_summary.get('worst_score')}")
    lines.append(f"- pass_rate: {baseline_summary.get('pass_rate')}")
    lines.append(f"- worst_corner_id: {baseline_summary.get('worst_corner_id')}")
    for name, mv in baseline_metrics.values.items():
        lines.append(format_metric_line(name, mv))
    if param_value_errors:
        lines.append("")
        lines.append("## Param Value Errors")
        for err in param_value_errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("## Top-K Candidates (Worst Score)")
    for entry in top_k(candidate_entries, int(corner_cfg.get("top_k", 5))):
        lines.append(
            f"- iter {entry.get('iteration')}: worst_score={entry.get('worst_score')} "
            f"pass_rate={entry.get('pass_rate')} worst_corner_id={entry.get('worst_corner_id')} "
            f"nominal={_format_metrics(entry.get('nominal_metrics') or {})} "
            f"worst={_format_metrics(entry.get('worst_metrics') or {})} "
            f"best={_format_metrics(entry.get('best_metrics') or {})}"
        )

    lines.append("")
    lines.append("## Pareto Front (Robust Losses)")
    for entry in pareto_entries:
        lines.append(
            f"- iter {entry.get('iteration')}: losses={entry.get('robust_losses')} "
            f"worst_score={entry.get('worst_score')} worst_corner_id={entry.get('worst_corner_id')} "
            f"nominal={_format_metrics(entry.get('nominal_metrics') or {})} "
            f"worst={_format_metrics(entry.get('worst_metrics') or {})} "
            f"best={_format_metrics(entry.get('best_metrics') or {})}"
        )

    lines.append("")
    lines.append("## Failures")
    lines.append(f"- sim_runs_failed: {sim_runs_failed}")

    lines.append("")
    lines.append("## Files")
    lines.append("- search/corner_set.json")
    lines.append("- search/topk.json")
    lines.append("- search/pareto.json")
    lines.append("- history/iterations.jsonl")
    lines.append("- best/best.sp")
    lines.append("- best/best_metrics.json")
    return lines
