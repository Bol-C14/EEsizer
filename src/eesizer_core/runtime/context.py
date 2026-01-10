from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import time

from ..contracts.provenance import RunManifest


@dataclass
class RunContext:
    workspace_root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    env: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def run_dir(self) -> Path:
        d = self.workspace_root / "runs" / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def manifest(self) -> RunManifest:
        return RunManifest(run_id=self.run_id, workspace=self.run_dir(), seed=self.seed, env=self.env, notes=self.notes)
