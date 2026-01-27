from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import CircuitSpec
from ..contracts.errors import ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json
from ..contracts.hashes import spec_payload, hash_spec


def _objective_map(spec: CircuitSpec) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in spec.objectives:
        out[str(obj.metric).strip()] = {
            "metric": obj.metric,
            "target": obj.target,
            "tol": obj.tol,
            "weight": obj.weight,
            "sense": obj.sense,
        }
    return out


def _constraints_payload(spec: CircuitSpec) -> list[dict[str, Any]]:
    items = []
    for c in spec.constraints:
        items.append({"kind": c.kind, "data": dict(c.data)})
    return sorted(items, key=lambda d: (str(d.get("kind", "")).lower(), stable_hash_json(d)))


def diff_specs(old: CircuitSpec, new: CircuitSpec) -> dict[str, Any]:
    old_obj = _objective_map(old)
    new_obj = _objective_map(new)

    added = sorted([m for m in new_obj.keys() if m not in old_obj], key=str.lower)
    removed = sorted([m for m in old_obj.keys() if m not in new_obj], key=str.lower)

    changed: list[dict[str, Any]] = []
    for metric in sorted(set(old_obj.keys()) & set(new_obj.keys()), key=str.lower):
        if old_obj[metric] != new_obj[metric]:
            changed.append({"metric": metric, "from": old_obj[metric], "to": new_obj[metric]})

    old_constraints = _constraints_payload(old)
    new_constraints = _constraints_payload(new)

    old_obs = set(old.observables)
    new_obs = set(new.observables)

    old_notes = dict(old.notes)
    new_notes = dict(new.notes)

    notes_changed_keys = sorted(
        {k for k in old_notes.keys() ^ new_notes.keys()} | {k for k in old_notes.keys() & new_notes.keys() if old_notes.get(k) != new_notes.get(k)},
        key=str,
    )
    sim_plan_changed = old_notes.get("sim_plan") != new_notes.get("sim_plan")

    return {
        "objectives_added": [new_obj[m] for m in added],
        "objectives_removed": [old_obj[m] for m in removed],
        "objectives_changed": changed,
        "constraints_changed": old_constraints != new_constraints,
        "observables_added": sorted(new_obs - old_obs),
        "observables_removed": sorted(old_obs - new_obs),
        "notes_changed_keys": notes_changed_keys,
        "sim_plan_changed": sim_plan_changed,
    }


@dataclass
class SpecDiffOperator(Operator):
    name: str = "spec_diff"
    version: str = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        old = inputs.get("old_spec") or inputs.get("a") or inputs.get("spec_a")
        new = inputs.get("new_spec") or inputs.get("b") or inputs.get("spec_b")
        if not isinstance(old, CircuitSpec) or not isinstance(new, CircuitSpec):
            raise ValidationError("SpecDiffOperator requires old_spec/new_spec as CircuitSpec")

        diff = diff_specs(old, new)

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["old_spec"] = ArtifactFingerprint(sha256=hash_spec(old).split("sha256:", 1)[1])
        prov.inputs["new_spec"] = ArtifactFingerprint(sha256=hash_spec(new).split("sha256:", 1)[1])
        prov.outputs["diff"] = ArtifactFingerprint(sha256=stable_hash_json(diff))
        prov.finish()

        return OperatorResult(outputs={"diff": diff}, provenance=prov)

