"""Execution context utilities for agent pipelines."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional

from .config import RunPathLayout
from .messaging import Message, MessageBundle


class ArtifactKind(str, Enum):
    """Enumerates the canonical artifact categories tracked by agents."""

    GENERIC = "generic"
    NETLIST = "netlist"
    PLAN = "plan"
    SIMULATION = "simulation"
    LOG = "log"
    OPTIMIZATION = "optimization"


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata-rich pointer to files created during a run."""

    name: str
    path: Path
    kind: ArtifactKind = ArtifactKind.GENERIC
    description: str | None = None
    metadata: MutableMapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "path": str(self.path),
            "kind": self.kind.value,
        }
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class EnvironmentMetadata:
    """Describes the PVT conditions attached to the execution context."""

    corner: str | None = None
    supply_voltage: float | None = None
    temperature_c: float | None = None
    extra: MutableMapping[str, str] = field(default_factory=dict)

    def update(
        self,
        *,
        corner: str | None = None,
        supply_voltage: float | None = None,
        temperature_c: float | None = None,
        extra: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        if corner is not None:
            self.corner = corner
        if supply_voltage is not None:
            self.supply_voltage = supply_voltage
        if temperature_c is not None:
            self.temperature_c = temperature_c
        if extra:
            self.extra.update(extra)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "corner": self.corner,
            "supply_voltage": self.supply_voltage,
            "temperature_c": self.temperature_c,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(slots=True)
class ExecutionContext:
    """Shared runtime state passed across agent stages."""

    run_id: str
    working_dir: Path
    config_name: str
    paths: RunPathLayout | None = None
    messages: MessageBundle = field(default_factory=lambda: MessageBundle(messages=[]))
    metadata: MutableMapping[str, str] = field(default_factory=dict)
    artifacts: MutableMapping[str, ArtifactRecord] = field(default_factory=dict)
    netlist_path: Optional[Path] = None
    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)

    def log(self, message: Message) -> None:
        """Append a message to the conversation transcript."""

        self.messages.messages.append(message)

    def attach_artifact(
        self,
        name: str,
        path: Path,
        *,
        kind: ArtifactKind = ArtifactKind.GENERIC,
        description: str | None = None,
        metadata: Optional[MutableMapping[str, str]] = None,
    ) -> ArtifactRecord:
        """Record the location of an artifact generated during the run."""

        record = ArtifactRecord(
            name=name,
            path=path,
            kind=kind,
            description=description,
            metadata=dict(metadata or {}),
        )
        try:
            record.metadata.setdefault("relative_path", str(path.relative_to(self.working_dir)))
        except ValueError:
            pass
        self.artifacts[name] = record
        return record

    def list_artifacts(self, *, kind: ArtifactKind | None = None) -> Iterable[ArtifactRecord]:
        """Iterate over artifacts optionally filtered by kind."""

        if kind is None:
            return list(self.artifacts.values())
        return [record for record in self.artifacts.values() if record.kind == kind]

    def set_environment(
        self,
        *,
        corner: str | None = None,
        supply_voltage: float | None = None,
        temperature_c: float | None = None,
        extra: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        """Update the contextual PVT data shared across agent stages."""

        self.environment.update(
            corner=corner, supply_voltage=supply_voltage, temperature_c=temperature_c, extra=extra
        )


class ContextManager(AbstractContextManager[ExecutionContext]):
    """Context manager that prepares and tears down execution directories."""

    def __init__(
        self,
        run_id: str,
        base_dir: Path,
        config_name: str,
        *,
        create_dirs: bool = True,
        pre_run_hook: Callable[[ExecutionContext], None] | None = None,
        post_run_hook: Callable[[ExecutionContext], None] | None = None,
        path_layout: RunPathLayout | None = None,
    ):
        self.run_id = run_id
        self.base_dir = base_dir
        self.config_name = config_name
        self.create_dirs = create_dirs
        self._context: ExecutionContext | None = None
        self._pre_run_hook = pre_run_hook
        self._post_run_hook = post_run_hook
        self._path_layout = path_layout

    def __enter__(self) -> ExecutionContext:
        workdir = self._path_layout.run_dir if self._path_layout else self.base_dir / self.run_id
        if self.create_dirs:
            workdir.mkdir(parents=True, exist_ok=True)
            if self._path_layout:
                for child in (
                    self._path_layout.artifacts,
                    self._path_layout.logs,
                    self._path_layout.simulations,
                    self._path_layout.plans,
                ):
                    child.mkdir(parents=True, exist_ok=True)
        self._context = ExecutionContext(
            run_id=self.run_id,
            working_dir=workdir,
            config_name=self.config_name,
            paths=self._path_layout,
        )
        if self._pre_run_hook:
            self._pre_run_hook(self._context)
        return self._context

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._context and self._post_run_hook:
            self._post_run_hook(self._context)
        self._context = None
        return None

    @property
    def context(self) -> ExecutionContext:
        if not self._context:
            raise RuntimeError("Context is only available inside the with-statement scope.")
        return self._context
