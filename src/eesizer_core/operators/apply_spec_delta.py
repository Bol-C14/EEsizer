from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import CircuitSpec, Objective
from ..contracts.deltas import SpecDelta
from ..contracts.errors import ValidationError
from ..contracts.hashes import hash_spec
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json
from ..metrics.aliases import canonicalize_metric_name


def _merge_notes(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            nested = dict(out[k])
            nested.update(dict(v))
            out[k] = nested
        else:
            out[k] = v
    return out


def apply_spec_delta(spec: CircuitSpec, delta: SpecDelta) -> CircuitSpec:
    obj_list = list(spec.objectives)
    obj_by_metric = {canonicalize_metric_name(o.metric): o for o in obj_list}

    for op in delta.objectives:
        metric = canonicalize_metric_name(op.metric)
        if not metric:
            continue

        if op.op == "remove":
            obj_by_metric.pop(metric, None)
            continue

        if op.op == "add":
            if metric in obj_by_metric:
                continue
            value = op.value
            if not isinstance(value, Mapping):
                raise ValidationError(f"spec_delta add for '{metric}' requires dict value")
            obj_by_metric[metric] = Objective(
                metric=metric,
                target=value.get("target"),
                tol=value.get("tol"),
                weight=float(value.get("weight", 1.0)),
                sense=str(value.get("sense", "ge")),
            )
            continue

        cur = obj_by_metric.get(metric)
        if cur is None:
            # For target/weight/tol updates, ignore missing objectives.
            continue

        if op.op == "target":
            obj_by_metric[metric] = Objective(metric=cur.metric, target=op.value, tol=cur.tol, weight=cur.weight, sense=cur.sense)
        elif op.op == "weight":
            obj_by_metric[metric] = Objective(metric=cur.metric, target=cur.target, tol=cur.tol, weight=float(op.value), sense=cur.sense)
        elif op.op == "tol":
            obj_by_metric[metric] = Objective(metric=cur.metric, target=cur.target, tol=op.value, weight=cur.weight, sense=cur.sense)
        elif op.op == "sense":
            obj_by_metric[metric] = Objective(metric=cur.metric, target=cur.target, tol=cur.tol, weight=cur.weight, sense=str(op.value))

    # Preserve original order where possible.
    new_objectives: list[Objective] = []
    seen: set[str] = set()
    for obj in obj_list:
        mid = canonicalize_metric_name(obj.metric)
        if mid in obj_by_metric and mid not in seen:
            new_objectives.append(obj_by_metric[mid])
            seen.add(mid)
    for mid in sorted(obj_by_metric.keys(), key=str.lower):
        if mid not in seen:
            new_objectives.append(obj_by_metric[mid])

    notes = _merge_notes(spec.notes, delta.notes)
    return CircuitSpec(
        objectives=tuple(new_objectives),
        constraints=tuple(spec.constraints),
        observables=tuple(spec.observables),
        notes=notes,
    )


@dataclass
class ApplySpecDeltaOperator(Operator):
    name: str = "apply_spec_delta"
    version: str = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        spec = inputs.get("spec")
        if not isinstance(spec, CircuitSpec):
            raise ValidationError("ApplySpecDeltaOperator requires 'spec' CircuitSpec")
        delta_raw = inputs.get("spec_delta") or inputs.get("delta") or {}
        if isinstance(delta_raw, SpecDelta):
            delta = delta_raw
        elif isinstance(delta_raw, Mapping):
            delta = SpecDelta.from_dict(delta_raw)
        else:
            raise ValidationError("ApplySpecDeltaOperator requires spec_delta dict")

        new_spec = apply_spec_delta(spec, delta)

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["spec"] = ArtifactFingerprint(sha256=hash_spec(spec).split("sha256:", 1)[1])
        prov.inputs["spec_delta"] = ArtifactFingerprint(sha256=stable_hash_json(delta.to_dict()))
        prov.outputs["new_spec"] = ArtifactFingerprint(sha256=hash_spec(new_spec).split("sha256:", 1)[1])
        prov.finish()

        return OperatorResult(outputs={"spec": new_spec, "old_spec": spec, "spec_delta": delta.to_dict()}, provenance=prov)

