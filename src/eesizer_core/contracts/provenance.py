from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import time


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_str(s: str) -> str:
    return stable_hash_bytes(s.encode("utf-8", errors="ignore"))


@dataclass(frozen=True)
class ArtifactFingerprint:
    """Stable identity for an artifact's content."""
    sha256: str


@dataclass
class Provenance:
    """Execution trace for an operator call."""
    operator: str
    version: str = "0.0"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    inputs: Dict[str, ArtifactFingerprint] = field(default_factory=dict)
    outputs: Dict[str, ArtifactFingerprint] = field(default_factory=dict)
    command: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.end_time = time.time()

    def duration_s(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return max(0.0, self.end_time - self.start_time)


@dataclass
class RunManifest:
    """Per-run metadata for reproducibility and audits."""
    run_id: str
    workspace: Path
    seed: Optional[int] = None
    tool_versions: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        payload = {
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "seed": self.seed,
            "tool_versions": self.tool_versions,
            "env": self.env,
            "notes": self.notes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
