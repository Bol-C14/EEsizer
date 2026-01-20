from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..contracts.artifacts import CircuitSource
from ..contracts.enums import SourceKind
from ..contracts.errors import ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_str
from .artifacts import NetlistBundle


class CircuitSourceToNetlistBundleOperator(Operator):
    """Adapter to build a NetlistBundle from a CircuitSource for legacy callers."""

    name = "circuit_source_to_netlist_bundle"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        src = inputs.get("circuit_source")
        if not isinstance(src, CircuitSource):
            raise ValidationError("circuit_source must be provided as CircuitSource")
        if src.kind != SourceKind.spice_netlist:
            raise ValidationError("circuit_source must be a SPICE netlist")

        md = src.metadata or {}
        base_dir = md.get("base_dir") or "."
        include_files = md.get("include_files") or ()
        extra_search_paths = md.get("extra_search_paths") or ()

        if not isinstance(base_dir, (str, Path)):
            raise ValidationError("base_dir must be a string or Path if provided")
        if not isinstance(include_files, (list, tuple)):
            raise ValidationError("include_files must be a list/tuple of paths if provided")
        if not isinstance(extra_search_paths, (list, tuple)):
            raise ValidationError("extra_search_paths must be a list/tuple of paths if provided")

        include_files = tuple(include_files)
        extra_search_paths = tuple(extra_search_paths)
        for path in include_files:
            if not isinstance(path, (str, Path)):
                raise ValidationError("include_files entries must be strings or Paths")
        for path in extra_search_paths:
            if not isinstance(path, (str, Path)):
                raise ValidationError("extra_search_paths entries must be strings or Paths")

        bundle = NetlistBundle(
            text=src.text,
            base_dir=base_dir,
            include_files=include_files,
            extra_search_paths=extra_search_paths,
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["circuit_source"] = ArtifactFingerprint(sha256=stable_hash_str(src.text))
        provenance.outputs["netlist_bundle"] = ArtifactFingerprint(sha256=stable_hash_str(bundle.text))
        provenance.finish()

        return OperatorResult(outputs={"netlist_bundle": bundle}, provenance=provenance)
