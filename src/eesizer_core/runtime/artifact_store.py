from __future__ import annotations

"""A small, auditable blackboard for multi-stage / multi-agent workflows.

Design goals:
- Deterministic serialization + stable content hashing.
- Safe, run_dir-relative writes (no path traversal).
- Optional in-memory cache for passing rich Python objects between stages.

This is intentionally lightweight: it is *not* a general-purpose object DB.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

from ..contracts.errors import ValidationError
from ..contracts.provenance import stable_hash_json, stable_hash_str
from .recorder import RunRecorder


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_relpath(root: str, name: str, *, ext: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("artifact name must be a non-empty string")
    # Allow hierarchical names but forbid traversal / absolute paths.
    p = Path(name)
    if p.is_absolute() or p.drive:
        raise ValidationError("artifact name must be run-relative")
    if any(part == ".." for part in p.parts):
        raise ValidationError("artifact name must not contain '..'")
    # Normalize extension.
    suffix = ext if ext.startswith(".") else f".{ext}"
    if p.suffix != suffix:
        p = p.with_suffix(suffix)
    root_p = Path(root)
    return str(root_p / p)


def _jsonable(value: Any) -> Any:
    """Convert a value into a JSON-serializable payload.

    Mirrors the runtime recorder behavior to keep hashing + on-disk JSON stable.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        # dataclasses.asdict() performs deepcopy, which fails on mappingproxy.
        # Do a shallow field walk instead.
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # mappingproxy (and similar read-only mappings) should be treated as dict.
    try:
        from types import MappingProxyType

        if isinstance(value, MappingProxyType):
            return {str(k): _jsonable(v) for k, v in value.items()}
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


class ArtifactStore:
    """Run-scoped artifact store.

    Artifacts are written under `root_dir` (run-relative), with an index that records
    hashes and metadata. The store also keeps an in-memory copy to preserve rich
    Python objects during the current process.
    """

    def __init__(self, recorder: RunRecorder, *, root_dir: str = "orchestrator/artifacts") -> None:
        self.recorder = recorder
        self.root_dir = root_dir.rstrip("/")
        self._index: Dict[str, Dict[str, Any]] = {}
        self._mem: Dict[str, Any] = {}

    def put(
        self,
        name: str,
        artifact: Any,
        *,
        kind: str = "json",
        producer: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store an artifact under `name`.

        kind: "json" (default) or "text".
        Returns the index entry.
        """
        kind_l = str(kind or "json").lower()
        if kind_l not in {"json", "text"}:
            raise ValidationError("artifact kind must be 'json' or 'text'")

        rel_path = _safe_relpath(self.root_dir, name, ext="json" if kind_l == "json" else "txt")

        if kind_l == "json":
            payload = _jsonable(artifact)
            sha256 = stable_hash_json(payload)
            self.recorder.write_json(rel_path, payload)
        else:
            text = artifact if isinstance(artifact, str) else json.dumps(_jsonable(artifact), sort_keys=True)
            sha256 = stable_hash_str(text)
            self.recorder.write_text(rel_path, text)

        entry: Dict[str, Any] = {
            "name": name,
            "kind": kind_l,
            "path": rel_path,
            "sha256": sha256,
            "producer": producer,
            "created_at": _utc_iso(),
            "meta": dict(meta or {}),
        }
        self._index[name] = entry
        self._mem[name] = artifact
        return entry

    def get(self, name: str) -> Any:
        if name in self._mem:
            return self._mem[name]
        # Best-effort load from disk as JSON.
        entry = self._index.get(name)
        if entry is None:
            raise KeyError(name)
        path = self.recorder.run_dir / entry["path"]
        if entry.get("kind") == "text":
            return path.read_text(encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return name in self._index

    def entry(self, name: str) -> Dict[str, Any]:
        if name not in self._index:
            raise KeyError(name)
        return dict(self._index[name])

    def list_names(self) -> list[str]:
        return sorted(self._index.keys())

    def dump_index(self) -> Path:
        """Write index.json and return its path."""
        payload = {
            "root_dir": self.root_dir,
            "artifacts": [self._index[k] for k in sorted(self._index.keys())],
        }
        rel_path = str(Path(self.root_dir) / "index.json")
        return self.recorder.write_json(rel_path, payload)
