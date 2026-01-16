from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import time
import platform
import sys
from datetime import datetime, timezone

from ..contracts.provenance import RunManifest
from .recorder import RunRecorder


@dataclass
class RunContext:
    workspace_root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    env: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    _manifest: Optional[RunManifest] = field(default=None, init=False, repr=False)
    _recorder: Optional[RunRecorder] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    def run_dir(self) -> Path:
        d = self.workspace_root / "runs" / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d.resolve()

    def manifest(self) -> RunManifest:
        if self._manifest is None:
            started = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
            environment = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
            }
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
        if self._recorder is None:
            self._recorder = RunRecorder(self.run_dir())
        return self._recorder
