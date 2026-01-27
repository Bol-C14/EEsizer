from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import math
import time


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_str(s: str) -> str:
    return stable_hash_bytes(s.encode("utf-8", errors="ignore"))


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        # Strict JSON doesn't permit NaN/Infinity.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return str(obj)


def stable_hash_json(obj: Any) -> str:
    """Stable hash using JSON with sorted keys and normalized containers."""
    normalized = _to_jsonable(obj)
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return stable_hash_bytes(data)


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
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    tool_versions: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(
            {
                "run_id": self.run_id,
                "workspace": str(self.workspace),
                "seed": self.seed,
                "timestamp_start": self.timestamp_start,
                "timestamp_end": self.timestamp_end,
                "inputs": self.inputs,
                "environment": self.environment,
                "files": self.files,
                "result_summary": self.result_summary,
                "tool_versions": self.tool_versions,
                "env": self.env,
                "notes": self.notes,
            }
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.write_text(payload, encoding="utf-8")
