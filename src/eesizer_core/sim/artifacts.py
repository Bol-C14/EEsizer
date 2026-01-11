from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, MutableMapping

from ..contracts.enums import SimKind


@dataclass(frozen=True)
class SpiceDeck:
    """Deck materialized for a single ngspice analysis."""

    text: str
    kind: SimKind
    # Mapping of logical name -> relative output file path expected from this deck.
    expected_outputs: Mapping[str, str] = field(default_factory=dict)
    # Mapping of logical name -> ordered column definitions for wrdata output.
    expected_outputs_meta: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Optional working directory for ngspice (for resolving .include paths).
    workdir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_outputs", MappingProxyType(dict(self.expected_outputs)))
        object.__setattr__(
            self,
            "expected_outputs_meta",
            MappingProxyType({k: tuple(v) for k, v in dict(self.expected_outputs_meta).items()}),
        )
        if self.workdir is not None:
            object.__setattr__(self, "workdir", Path(self.workdir))


@dataclass
class RawSimData:
    """Filesystem-backed record of an ngspice run."""

    kind: SimKind
    run_dir: Path
    outputs: MutableMapping[str, Path]
    log_path: Path
    cmdline: list[str]
    returncode: int
    outputs_meta: MutableMapping[str, tuple[str, ...]] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.log_path = Path(self.log_path)
        self.outputs = {name: Path(path) for name, path in dict(self.outputs).items()}
        self.outputs_meta = {name: tuple(cols) for name, cols in dict(self.outputs_meta).items()}
        self.cmdline = list(self.cmdline)

    def output_path(self, name: str) -> Path:
        """Helper to fetch an output path by logical name."""
        return self.outputs[name]


@dataclass(frozen=True)
class NetlistBundle:
    """Netlist text + base_dir for resolving includes."""

    text: str
    base_dir: Path
    include_files: tuple[Path, ...] = ()
    extra_search_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_dir", Path(self.base_dir))
        object.__setattr__(self, "include_files", tuple(Path(p) for p in self.include_files))
        object.__setattr__(self, "extra_search_paths", tuple(Path(p) for p in self.extra_search_paths))
