"""Netlist helpers used by the CLI pipeline and agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


IGNORED_PREFIXES = ("*", ".")


@dataclass(slots=True)
class NetlistData:
    """Container that keeps both the raw text and computed summary."""

    path: Path
    text: str

    @property
    def component_counts(self) -> Mapping[str, int]:
        return summarize_components(self.text)


@dataclass(slots=True)
class NetlistSummary:
    """Lightweight metadata extracted from a SPICE netlist."""

    path: Path
    component_counts: Mapping[str, int]
    total_components: int
    unique_nodes: int

    def describe(self) -> str:
        ordered = sorted(self.component_counts.items())
        pairs = ", ".join(f"{kind}:{count}" for kind, count in ordered)
        return (
            f"Netlist '{self.path.name}' contains {self.total_components} elements "
            f"across {self.unique_nodes} unique nodes ({pairs})."
        )


def load_netlist(path: Path) -> NetlistData:
    """Read a SPICE netlist from disk."""

    text = path.read_text()
    return NetlistData(path=path, text=text)


def summarize_components(text: str) -> Mapping[str, int]:
    """Count component types by their first-letter prefix."""

    counts: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(IGNORED_PREFIXES):
            continue
        symbol = stripped[0].upper()
        counts[symbol] += 1
    return counts


def summarize_netlist(path: Path, text: str) -> NetlistSummary:
    """Return basic metadata about the provided netlist text."""

    counts = summarize_components(text)
    total_components = sum(counts.values())
    nodes = _collect_nodes(text)
    return NetlistSummary(
        path=path,
        component_counts=dict(counts),
        total_components=total_components,
        unique_nodes=len(nodes),
    )


def _collect_nodes(text: str) -> set[str]:
    nodes: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(IGNORED_PREFIXES):
            continue
        tokens = stripped.split()
        # For most SPICE components the next three tokens reference nodes.
        for token in tokens[1:4]:
            if not token or token.startswith("$"):
                continue
            if token[0].isalpha() or token[0].isdigit():
                nodes.add(token)
    return nodes


__all__ = [
    "IGNORED_PREFIXES",
    "NetlistData",
    "NetlistSummary",
    "load_netlist",
    "summarize_components",
    "summarize_netlist",
]
