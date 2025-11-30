"""Utility helpers for building ngspice control decks and parsing results."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class SimulationDirective:
    """Represents a single analysis statement inside a ``.control`` block."""

    label: str
    statement: str
    description: str | None = None


@dataclass(slots=True)
class MeasurementSpec:
    """Describes a ``.measure`` statement with optional docstring."""

    name: str
    statement: str
    analysis: str = "ac"
    description: str | None = None


@dataclass(slots=True)
class WaveformExport:
    """Represents a ``wrdata`` command used to dump saved vectors."""

    filename: str
    vectors: Sequence[str]
    plot: str | None = None
    description: str | None = None


@dataclass(slots=True)
class ControlDeck:
    """Container that holds the directives and measurements for a run."""

    name: str
    directives: Sequence[SimulationDirective] = field(default_factory=tuple)
    measurements: Sequence[MeasurementSpec] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)
    saves: Sequence[str] = field(default_factory=tuple)
    waveform_exports: Sequence[WaveformExport] = field(default_factory=tuple)

    def render(self) -> str:
        """Return a block of analysis/measurement statements."""

        lines: list[str] = [".control"]
        if self.saves:
            unique = []
            for target in self.saves:
                if target and target not in unique:
                    unique.append(target)
            if unique:
                lines.append("* --- saved waveforms ---")
                lines.append("save " + " ".join(unique))
        lines.append("* --- simulation directives ---")
        for directive in self.directives:
            if directive.description:
                lines.append(f"* {directive.description}")
            lines.append(directive.statement)
        if self.measurements:
            lines.append("* --- measurements ---")
            for measurement in self.measurements:
                if measurement.description:
                    lines.append(f"* {measurement.description}")
                lines.append(measurement.statement)
        if self.notes:
            lines.append("* --- notes ---")
            lines.extend(self.notes)
        lines.append("run")
        if self.waveform_exports:
            lines.append("* --- waveform exports ---")
            lines.append("set filetype=ascii")
            for export in self.waveform_exports:
                if export.description:
                    lines.append(f"* {export.description}")
                if export.plot:
                    lines.append(f"setplot {export.plot}")
                vector_list = " ".join(export.vectors)
                lines.append(f"wrdata {export.filename} {vector_list}")
        lines.append(".endc")
        return "\n".join(lines) + "\n"


def augment_netlist(netlist_text: str, deck: ControlDeck) -> str:
    """Append the rendered control deck to a SPICE netlist."""

    base_text = netlist_text.rstrip("\n")
    deck_text = deck.render().strip("\n")
    if not deck_text:
        return base_text + "\n"

    base_lines = base_text.splitlines()
    deck_lines = deck_text.splitlines()
    insert_idx = None
    for idx in range(len(base_lines) - 1, -1, -1):
        token = base_lines[idx].strip().lower()
        if token.startswith(".end") and not token.startswith(".endc"):
            insert_idx = idx
            break

    if insert_idx is None:
        combined_lines = base_lines + [""] + deck_lines + ["", ".end"]
    else:
        combined_lines = (
            base_lines[:insert_idx]
            + [""]
            + deck_lines
            + [""]
            + base_lines[insert_idx:]
        )

    return "\n".join(combined_lines).rstrip("\n") + "\n"


def dc_simulation(source: str, start: float, stop: float, step: float, *, label: str | None = None,
                  description: str | None = None) -> SimulationDirective:
    label = label or f"dc_{source.lower()}"
    statement = f".dc {source} {start:g} {stop:g} {step:g}"
    return SimulationDirective(label=label, statement=statement, description=description)


def ac_simulation(sweep: str, points: int, start_hz: float, stop_hz: float, *,
                  label: str = "ac_sweep", description: str | None = None) -> SimulationDirective:
    statement = f".ac {sweep} {points} {start_hz:g} {stop_hz:g}"
    return SimulationDirective(label=label, statement=statement, description=description)


def tran_simulation(step: float, stop: float, *, start: float = 0.0, label: str = "tran", description: str | None = None) -> SimulationDirective:
    statement = f".tran {step:g} {stop:g} {start:g}"
    return SimulationDirective(label=label, statement=statement, description=description)


def measure_gain(name: str, output_node: str, input_node: str, *, analysis: str = "ac",
                 description: str | None = None) -> MeasurementSpec:
    expr = f".measure {analysis} {name} param='20*log10(abs(v({output_node})/v({input_node})))'"
    return MeasurementSpec(name=name, statement=expr, analysis=analysis, description=description)


def measure_voltage(name: str, node: str, *, analysis: str = "dc", description: str | None = None) -> MeasurementSpec:
    expr = f".measure {analysis} {name} find v({node})"
    return MeasurementSpec(name=name, statement=expr, analysis=analysis, description=description)


def measure_thd(name: str, node: str, fundamental_hz: float, *, description: str | None = None) -> MeasurementSpec:
    expr = f".measure tran {name} thd v({node}) {fundamental_hz:g}"
    return MeasurementSpec(name=name, statement=expr, analysis="tran", description=description)


def measure_power(name: str, supply_source: str, *, description: str | None = None) -> MeasurementSpec:
    expr = f".measure tran {name} param='-AVG(v({supply_source})*i({supply_source})) * 1e3'"
    return MeasurementSpec(name=name, statement=expr, analysis="tran", description=description)


MEASURE_LINE = re.compile(r"^\s*Measure\s+(?P<name>[\w\-]+)\s*=\s*(?P<value>[-+eE0-9\.]+)")
ALT_MEASURE = re.compile(r"^\s*(?P<name>[\w\-]+)\s*=\s*(?P<value>[-+eE0-9\.]+)")


def parse_measure_log(text: str) -> MutableMapping[str, float]:
    """Extract measurement values from an ngspice log file."""

    metrics: MutableMapping[str, float] = {}
    for line in text.splitlines():
        match = MEASURE_LINE.search(line) or ALT_MEASURE.search(line)
        if not match:
            continue
        name = match.group("name").strip()
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue
        metrics[name] = value
    return metrics


def load_measurements(path: Path) -> MutableMapping[str, float]:
    """Read measurements from a file that might be JSON or raw log text."""

    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return parse_measure_log(text)
    if isinstance(payload, Mapping):
        return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
    return parse_measure_log(text)


def describe_measurements(metrics: Mapping[str, float]) -> str:
    """Return a deterministic summary string used by the CLI run log."""

    if not metrics:
        return "No measurements available"
    ordered = sorted(metrics.items())
    return ", ".join(f"{key}={value:.3f}" for key, value in ordered)


__all__ = [
    "SimulationDirective",
    "MeasurementSpec",
    "WaveformExport",
    "ControlDeck",
    "augment_netlist",
    "dc_simulation",
    "ac_simulation",
    "tran_simulation",
    "measure_gain",
    "measure_voltage",
    "measure_power",
    "measure_thd",
    "parse_measure_log",
    "load_measurements",
    "describe_measurements",
]
