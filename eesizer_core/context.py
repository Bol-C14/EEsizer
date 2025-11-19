"""Execution context utilities for agent pipelines."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, MutableMapping, Optional

from .messaging import Message, MessageBundle


@dataclass(slots=True)
class ExecutionContext:
    """Shared runtime state passed across agent stages."""

    run_id: str
    working_dir: Path
    config_name: str
    messages: MessageBundle = field(default_factory=lambda: MessageBundle(messages=[]))
    metadata: MutableMapping[str, str] = field(default_factory=dict)
    artifacts: MutableMapping[str, Path] = field(default_factory=dict)
    netlist_path: Optional[Path] = None

    def log(self, message: Message) -> None:
        """Append a message to the conversation transcript."""

        self.messages.messages.append(message)

    def attach_artifact(self, name: str, path: Path) -> None:
        """Record the location of an artifact generated during the run."""

        self.artifacts[name] = path


class ContextManager(AbstractContextManager[ExecutionContext]):
    """Context manager that prepares and tears down execution directories."""

    def __init__(self, run_id: str, base_dir: Path, config_name: str, create_dirs: bool = True):
        self.run_id = run_id
        self.base_dir = base_dir
        self.config_name = config_name
        self.create_dirs = create_dirs
        self._context: ExecutionContext | None = None

    def __enter__(self) -> ExecutionContext:
        workdir = self.base_dir / self.run_id
        if self.create_dirs:
            workdir.mkdir(parents=True, exist_ok=True)
        self._context = ExecutionContext(
            run_id=self.run_id,
            working_dir=workdir,
            config_name=self.config_name,
        )
        return self._context

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        # Placeholder for cleanup hooks such as flushing transcripts or archiving artifacts.
        self._context = None
        return None

    @property
    def context(self) -> ExecutionContext:
        if not self._context:
            raise RuntimeError("Context is only available inside the with-statement scope.")
        return self._context
