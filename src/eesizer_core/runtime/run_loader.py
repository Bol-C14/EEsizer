from __future__ import annotations

"""Utilities for reading recorded run artifacts from disk."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import json


@dataclass
class RunLoader:
    """Load artifacts from a run directory created by the runtime recorder."""

    run_dir: Path

    def __post_init__(self) -> None:
        # Normalize to a Path in case callers pass in strings.
        self.run_dir = Path(self.run_dir)

    def load_manifest(self) -> dict[str, Any]:
        """Return the run manifest JSON, or an empty dict if missing."""
        manifest_path = self.run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def iter_history(self) -> Iterator[dict[str, Any]]:
        """Yield history rows from the JSONL history file, or an empty iterator."""
        history_path = self.run_dir / "history" / "iterations.jsonl"
        if not history_path.exists():
            return iter(())

        def _iter() -> Iterator[dict[str, Any]]:
            with history_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    yield json.loads(line)

        return _iter()

    def load_best(self) -> dict[str, Any]:
        """Return the recorded best netlist text and metrics (if present)."""
        best_dir = self.run_dir / "best"
        best_sp = best_dir / "best.sp"
        best_metrics = best_dir / "best_metrics.json"
        return {
            "best_sp": best_sp.read_text(encoding="utf-8") if best_sp.exists() else "",
            "best_metrics": json.loads(best_metrics.read_text(encoding="utf-8")) if best_metrics.exists() else {},
        }


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Convenience wrapper around RunLoader.load_manifest()."""
    return RunLoader(run_dir).load_manifest()


def iter_history(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Convenience wrapper around RunLoader.iter_history()."""
    return RunLoader(run_dir).iter_history()


def load_best(run_dir: Path) -> dict[str, Any]:
    """Convenience wrapper around RunLoader.load_best()."""
    return RunLoader(run_dir).load_best()
