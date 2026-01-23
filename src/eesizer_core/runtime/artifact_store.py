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
import importlib
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


def _type_tag(value: Any) -> str | None:
    if is_dataclass(value):
        return f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return None


def _resolve_type(tag: str) -> type | None:
    if not tag:
        return None
    try:
        module_name, _, qualname = tag.rpartition(".")
        if not module_name:
            return None
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if isinstance(obj, type):
            return obj
    except Exception:
        return None
    return None


def _encode_artifact(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(k): _encode_artifact(v) for k, v in value.items()}
    try:
        from types import MappingProxyType

        if isinstance(value, MappingProxyType):
            return {str(k): _encode_artifact(v) for k, v in value.items()}
    except Exception:
        pass
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_artifact(v) for v in value]}
    if isinstance(value, (list, set)):
        return [_encode_artifact(v) for v in value]
    if is_dataclass(value):
        data = {f.name: _encode_artifact(getattr(value, f.name)) for f in fields(value)}
        return {"__type__": _type_tag(value), "data": data}
    try:
        from enum import Enum

        if isinstance(value, Enum):
            return {
                "__enum__": _type_tag(value.__class__) or value.__class__.__name__,
                "value": _encode_artifact(value.value),
            }
    except Exception:
        pass
    return str(value)


def _decode_artifact(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_artifact(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__tuple__" in value:
        return tuple(_decode_artifact(v) for v in value.get("__tuple__", []))
    if "__bytes__" in value:
        try:
            return bytes.fromhex(str(value.get("__bytes__", "")))
        except Exception:
            return value
    if "__path__" in value:
        return Path(str(value.get("__path__", "")))
    if "__enum__" in value:
        tag = str(value.get("__enum__", ""))
        enum_cls = _resolve_type(tag)
        enum_value = _decode_artifact(value.get("value"))
        if enum_cls is None:
            return enum_value
        try:
            return enum_cls(enum_value)
        except Exception:
            return enum_value
    if "__type__" in value:
        tag = str(value.get("__type__", ""))
        data = value.get("data", {})
        cls = _resolve_type(tag)
        if cls is None or not is_dataclass(cls):
            return _decode_artifact(data)
        if not isinstance(data, dict):
            return _decode_artifact(data)
        kwargs = {k: _decode_artifact(v) for k, v in data.items()}
        if getattr(cls, "__name__", "") == "ParamSpace":
            params = kwargs.get("params")
            if isinstance(params, (list, tuple)):
                try:
                    return cls.build(list(params))
                except Exception:
                    return cls(params=tuple(params))
        try:
            return cls(**kwargs)
        except Exception:
            return kwargs
    return {str(k): _decode_artifact(v) for k, v in value.items()}


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

        type_tag = _type_tag(artifact)

        if kind_l == "json":
            payload = _encode_artifact(artifact)
            sha256 = stable_hash_json(payload)
            self.recorder.write_json(rel_path, payload)
        else:
            text = artifact if isinstance(artifact, str) else json.dumps(_encode_artifact(artifact), sort_keys=True)
            sha256 = stable_hash_str(text)
            self.recorder.write_text(rel_path, text)

        entry: Dict[str, Any] = {
            "name": name,
            "kind": kind_l,
            "path": rel_path,
            "sha256": sha256,
            "type_tag": type_tag,
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _decode_artifact(payload)

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

    def load_index(self, path: Path | None = None) -> None:
        """Load index.json from disk to repopulate the store."""
        if path is None:
            path = self.recorder.run_dir / self.root_dir / "index.json"
        if not Path(path).exists():
            raise FileNotFoundError(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        root_dir = payload.get("root_dir")
        if isinstance(root_dir, str) and root_dir:
            self.root_dir = root_dir.rstrip("/")
        artifacts = payload.get("artifacts", [])
        index: Dict[str, Dict[str, Any]] = {}
        for entry in artifacts:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            index[name] = dict(entry)
        self._index = index
        self._mem = {}
