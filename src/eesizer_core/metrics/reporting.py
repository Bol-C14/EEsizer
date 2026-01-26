from __future__ import annotations

from typing import Iterable

from ..contracts.artifacts import MetricValue
from ..contracts.metrics import METRIC_DEFINITIONS


def metric_definition_lines(metric_names: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for name in metric_names:
        if name not in seen:
            seen.append(name)
    lines: list[str] = []
    definitions = [METRIC_DEFINITIONS.get(name) for name in seen if name in METRIC_DEFINITIONS]
    if not definitions:
        return lines
    lines.append("## Metric Definitions")
    for definition in definitions:
        req = f" Requirements: {definition.requirements}" if definition and definition.requirements else ""
        if definition is None:
            continue
        lines.append(f"- {definition.name} ({definition.unit}): {definition.definition}{req}")
    return lines


def format_metric_line(name: str, metric: MetricValue) -> str:
    if metric.value is None:
        reason = metric.details.get("reason")
        status = metric.details.get("status", "missing")
        if reason:
            return f"- metric {name}: {status} ({reason})"
        return f"- metric {name}: {status}"
    return f"- metric {name}: {metric.value} {metric.unit or ''}".rstrip()
