from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import os


class RunRecorder:
    """Lightweight writer for run artifacts and JSONL logs."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    def relpath(self, path: str | Path) -> str:
        p = Path(path)
        try:
            rel = p.resolve().relative_to(self.run_dir)
            return str(rel)
        except Exception:
            return str(path)

    def _jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
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
        if os.path.isabs(value):
            try:
                return self.relpath(value)
            except Exception:
                return value
        return value

    def _normalize_payload(self, payload: Any) -> Any:
        value = self._jsonable(payload)
        if isinstance(value, str):
            return self._maybe_relpath_str(value)
        if isinstance(value, dict):
            return {k: self._normalize_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_payload(v) for v in value]
        return value

    def _write_text(self, rel_path: str, text: str) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_input(self, name: str, payload: Any) -> Path:
        rel_path = f"inputs/{name}"
        if isinstance(payload, str):
            return self._write_text(rel_path, payload)
        return self.write_json(rel_path, payload)

    def write_text(self, rel_path: str, text: str) -> Path:
        return self._write_text(rel_path, text)

    def write_json(self, rel_path: str, payload: Any) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_payload(payload)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def append_jsonl(self, rel_path: str, payload: Any) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_payload(payload)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(normalized, sort_keys=True))
            fh.write("\n")
        return path
