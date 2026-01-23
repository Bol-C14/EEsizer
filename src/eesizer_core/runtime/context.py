"""Run-scoped context used by operators/strategies to persist metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import time
import platform
import sys
from datetime import datetime, timezone
import importlib

from ..contracts.provenance import RunManifest
from .recorder import RunRecorder


@dataclass
class RunContext:
    """Holds run metadata and provides access to run storage helpers."""

    workspace_root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    env: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    _manifest: Optional[RunManifest] = field(default=None, init=False, repr=False)
    _recorder: Optional[RunRecorder] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve early so downstream paths are stable.
        self.workspace_root = Path(self.workspace_root).resolve()

    def run_dir(self) -> Path:
        """Return (and create) the run directory for this context."""
        d = self.workspace_root / "runs" / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d.resolve()

    def manifest(self) -> RunManifest:
        """Return the lazily-built RunManifest for this run."""
        if self._manifest is None:
            started = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
            environment = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
            }
            dep_snapshot = _dependency_snapshot()
            if dep_snapshot:
                environment["dependency_snapshot"] = dep_snapshot
            self._manifest = RunManifest(
                run_id=self.run_id,
                workspace=self.run_dir(),
                seed=self.seed,
                timestamp_start=started,
                environment=environment,
                env=self.env,
                notes=self.notes,
            )
        return self._manifest

    def recorder(self) -> RunRecorder:
        """Return a RunRecorder bound to this run's directory."""
        if self._recorder is None:
            self._recorder = RunRecorder(self.run_dir())
        return self._recorder


def _dependency_snapshot() -> list[str]:
    """Collect a best-effort dependency snapshot for reproducibility."""
    try:
        metadata = importlib.import_module("importlib.metadata")
    except Exception:
        return []
    try:
        distributions = metadata.distributions()
    except Exception:
        return []

    entries: list[str] = []
    for dist in distributions:
        try:
            name = dist.metadata.get("Name")
            version = dist.version
        except Exception:
            continue
        if not name or not version:
            continue
        entries.append(f"{name}=={version}")
    return sorted(set(entries), key=str.lower)
