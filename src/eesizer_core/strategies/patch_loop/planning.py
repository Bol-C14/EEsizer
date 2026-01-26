from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamSpace, SimPlan, SimRequest
from ...contracts.enums import SimKind
from ...contracts.policy import Observation
from ...metrics import MetricRegistry


def group_metric_names_by_kind(registry: MetricRegistry, metric_names: Iterable[str]) -> dict[SimKind, list[str]]:
    grouped: dict[SimKind, list[str]] = {}
    specs = registry.resolve(metric_names)
    for spec in specs:
        grouped.setdefault(spec.requires_kind, []).append(spec.name)
    return grouped


def sim_plan_for_kind(kind: SimKind) -> SimPlan:
    return SimPlan(sims=(SimRequest(kind=kind, params={}),))


def extract_sim_plan(notes: Mapping[str, Any]) -> SimPlan | None:
    raw = notes.get("sim_plan")
    if isinstance(raw, SimPlan):
        return raw
    if isinstance(raw, Mapping):
        sims_raw = raw.get("sims")
        if not isinstance(sims_raw, list) or not sims_raw:
            return None
        sims: list[SimRequest] = []
        for item in sims_raw:
            if not isinstance(item, Mapping):
                return None
            kind = item.get("kind")
            params = item.get("params", {})
            if not isinstance(kind, str) or not isinstance(params, Mapping):
                return None
            try:
                sim_kind = SimKind(kind)
            except ValueError:
                return None
            sims.append(SimRequest(kind=sim_kind, params=dict(params)))
        return SimPlan(sims=tuple(sims))
    return None


def merge_metrics(bundles: Iterable[MetricsBundle]) -> MetricsBundle:
    out = MetricsBundle()
    for bundle in bundles:
        out.values.update(bundle.values)
    return out


def make_observation(
    spec: CircuitSpec,
    source: CircuitSource,
    param_space: ParamSpace,
    metrics: MetricsBundle,
    iteration: int,
    history: list[dict[str, Any]],
    history_tail_k: int,
    notes: Mapping[str, Any],
) -> Observation:
    tail = history[-history_tail_k:] if history_tail_k > 0 else []
    return Observation(
        spec=spec,
        source=source,
        param_space=param_space,
        metrics=metrics,
        iteration=iteration,
        history_tail=tail,
        notes=dict(notes),
    )
