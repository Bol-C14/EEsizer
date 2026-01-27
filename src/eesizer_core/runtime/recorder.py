"""Record run inputs/outputs to disk in JSON or JSONL form."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import os


class RunRecorder:
    """Lightweight writer for run artifacts and JSONL logs."""

    def __init__(self, run_dir: Path) -> None:
        """Bind the recorder to a run directory on disk."""
        self.run_dir = Path(run_dir).resolve()

    def relpath(self, path: str | Path) -> str:
        """Return a run_dir-relative path string when possible."""
        p = Path(path)
        try:
            rel = p.resolve().relative_to(self.run_dir)
            return str(rel)
        except Exception:
            return str(path)

    def _jsonable(self, value: Any) -> Any:
        """Convert a value into a JSON-serializable structure."""
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            # Strict JSON doesn't permit NaN/Infinity.
            return value if math.isfinite(value) else None
        if isinstance(value, Path):
            return self.relpath(value)
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(v) for v in value]
        if is_dataclass(value):
            return self._jsonable(asdict(value))
        if isinstance(value, Iterable):
            return [self._jsonable(v) for v in value]
        return str(value)

    def _maybe_relpath_str(self, value: str) -> str:
        """Normalize absolute path strings to run_dir-relative where possible."""
        if os.path.isabs(value):
            try:
                return self.relpath(value)
            except Exception:
                return value
        return value

    def _normalize_payload(self, payload: Any) -> Any:
        """Normalize nested payloads, including path-like strings."""
        value = self._jsonable(payload)
        if isinstance(value, str):
            return self._maybe_relpath_str(value)
        if isinstance(value, dict):
            return {k: self._normalize_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_payload(v) for v in value]
        return value

    def _write_text(self, rel_path: str, text: str) -> Path:
        """Write UTF-8 text to a path under run_dir."""
        path = self._resolve_rel_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_input(self, name: str, payload: Any) -> Path:
        """Write inputs under inputs/, picking text or JSON based on payload type."""
        rel_path = f"inputs/{name}"
        if isinstance(payload, str):
            return self._write_text(rel_path, payload)
        return self.write_json(rel_path, payload)

    def write_text(self, rel_path: str, text: str) -> Path:
        """Write raw text to a run-relative path."""
        return self._write_text(rel_path, text)

    def write_json(self, rel_path: str, payload: Any) -> Path:
        """Serialize payload to JSON at a run-relative path."""
        path = self._resolve_rel_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_payload(payload)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        return path

    def append_jsonl(self, rel_path: str, payload: Any) -> Path:
        """Append one JSON record (per line) to a run-relative path."""
        path = self._resolve_rel_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_payload(payload)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(normalized, sort_keys=True, allow_nan=False))
            fh.write("\n")
        return path

    def _resolve_rel_path(self, rel_path: str) -> Path:
        """Resolve a safe run-relative path and block traversal."""
        path = Path(rel_path)
        if path.is_absolute() or path.drive:
            raise ValueError("rel_path must be relative to run_dir")
        if any(part == ".." for part in path.parts):
            raise ValueError("rel_path must not contain '..'")
        resolved = (self.run_dir / path).resolve()
        try:
            resolved.relative_to(self.run_dir)
        except Exception as exc:
            raise ValueError("rel_path escapes run_dir") from exc
        return resolved
