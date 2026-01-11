from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from ..contracts.errors import SimulationError, ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ..contracts.enums import SimKind
from ..runtime.context import RunContext
from .artifacts import RawSimData, SpiceDeck


def _tail(text: str, limit: int = 2000) -> str:
    """Return the trailing slice of text for quick diagnostics."""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _normalize_expected_outputs(expected: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, rel_path in expected.items():
        if not isinstance(key, str):
            raise ValidationError("expected_outputs keys must be strings")
        if not isinstance(rel_path, str):
            raise ValidationError("expected_outputs values must be relative path strings")
        normalized[key] = rel_path
    return normalized


class NgspiceRunOperator(Operator):
    """Run ngspice in batch mode given a prepared deck."""

    name = "ngspice_run"
    version = "0.1.0"

    def __init__(self, ngspice_bin: str = "ngspice", timeout_s: Optional[float] = None) -> None:
        self.ngspice_bin = ngspice_bin
        self.timeout_s = timeout_s

    def _write_deck(self, deck: SpiceDeck, stage_dir: Path) -> Path:
        deck_path = stage_dir / f"deck_{deck.kind.value}.sp"
        deck_text = deck.text
        # Ensure wrdata targets land in stage_dir even if ngspice cwd differs.
        for rel_path in deck.expected_outputs.values():
            abs_path = stage_dir / rel_path
            deck_text = deck_text.replace(rel_path, str(abs_path))
        deck_path.write_text(deck_text, encoding="utf-8")
        return deck_path

    def _log_path(self, deck: SpiceDeck, stage_dir: Path) -> Path:
        return stage_dir / f"ngspice_{deck.kind.value}.log"

    def _provenance_for_deck(self, deck: SpiceDeck) -> Provenance:
        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["deck_kind"] = ArtifactFingerprint(sha256=stable_hash_str(deck.kind.value))
        prov.inputs["deck_text"] = ArtifactFingerprint(sha256=stable_hash_str(deck.text))
        prov.inputs["expected_outputs"] = ArtifactFingerprint(
            sha256=stable_hash_json(sorted(deck.expected_outputs.items()))
        )
        return prov

    def run(self, inputs: Mapping[str, Any], ctx: Optional[RunContext]) -> OperatorResult:
        deck = inputs.get("deck")
        if not isinstance(deck, SpiceDeck):
            raise ValidationError("deck must be provided as a SpiceDeck")

        if not isinstance(deck.kind, SimKind):
            raise ValidationError("deck.kind must be a SimKind")

        expected_outputs = _normalize_expected_outputs(deck.expected_outputs)

        if ctx is None or not hasattr(ctx, "run_dir"):
            raise ValidationError("RunContext with run_dir() is required for ngspice runs")

        stage_name = inputs.get("stage") or deck.kind.value
        if not isinstance(stage_name, str):
            raise ValidationError("stage must be a string if provided")

        stage_dir = ctx.run_dir() / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        deck_path = self._write_deck(deck, stage_dir)
        log_path = self._log_path(deck, stage_dir)

        cmd = [self.ngspice_bin, "-b", "-o", str(log_path), str(deck_path)]

        provenance = self._provenance_for_deck(deck)
        provenance.command = " ".join(cmd)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_s,
                cwd=deck.workdir or stage_dir,
            )
        except FileNotFoundError as exc:
            raise SimulationError(f"ngspice executable not found: {self.ngspice_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SimulationError(f"ngspice timed out after {self.timeout_s} seconds") from exc

        stdout_tail = _tail(proc.stdout or "")
        stderr_tail = _tail(proc.stderr or "")

        if proc.returncode != 0:
            raise SimulationError(
                f"ngspice exited with code {proc.returncode}; see log at {log_path}\n"
                f"stderr tail: {stderr_tail or '<empty>'}"
            )

        outputs: dict[str, Path] = {}
        for key, rel_path in expected_outputs.items():
            out_path = stage_dir / rel_path
            if not out_path.exists():
                raise SimulationError(f"Expected output '{key}' not found at {out_path}")
            outputs[key] = out_path

        raw_data = RawSimData(
            kind=deck.kind,
            run_dir=stage_dir,
            outputs=outputs,
            outputs_meta=dict(deck.expected_outputs_meta),
            log_path=log_path,
            cmdline=cmd,
            returncode=proc.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

        provenance.outputs["raw_sim_data"] = ArtifactFingerprint(
            sha256=stable_hash_json(
                {
                    "kind": deck.kind.value,
                    "run_dir": str(stage_dir),
                    "deck_path": str(deck_path),
                    "log_path": str(log_path),
                    "outputs": {k: str(v) for k, v in outputs.items()},
                    "returncode": proc.returncode,
                }
            )
        )
        provenance.finish()

        return OperatorResult(
            outputs={
                "raw_data": raw_data,
                "deck_path": deck_path,
                "log_path": log_path,
            },
            provenance=provenance,
            logs={"stdout_tail": stdout_tail, "stderr_tail": stderr_tail},
        )
