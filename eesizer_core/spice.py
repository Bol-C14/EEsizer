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
class ControlDeck:
    """Container that holds the directives and measurements for a run."""

    name: str
    directives: Sequence[SimulationDirective] = field(default_factory=tuple)
    measurements: Sequence[MeasurementSpec] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)

    def render(self) -> str:
        """Return a ``.control`` block that ngspice understands."""

        lines = [".control"]
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
        lines.append("quit")
        lines.append(".endc")
        return "\n".join(lines) + "\n"


def augment_netlist(netlist_text: str, deck: ControlDeck) -> str:
    """Append the rendered control deck to a SPICE netlist."""

    text = netlist_text.rstrip() + "\n\n" + deck.render()
    if not text.strip().endswith(".end"):
        text = text.rstrip() + "\n.end\n"
    return text


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
    expr = f".measure {analysis} {name} param='20*log10(abs(V({output_node})/V({input_node})))'"
    return MeasurementSpec(name=name, statement=expr, analysis=analysis, description=description)


def measure_voltage(name: str, node: str, *, analysis: str = "dc", description: str | None = None) -> MeasurementSpec:
    expr = f".measure {analysis} {name} find V({node})"
    return MeasurementSpec(name=name, statement=expr, analysis=analysis, description=description)


def measure_thd(name: str, node: str, fundamental_hz: float, *, description: str | None = None) -> MeasurementSpec:
    expr = f".measure tran {name} thd V({node}) {fundamental_hz:g}"
    return MeasurementSpec(name=name, statement=expr, analysis="tran", description=description)


def measure_power(name: str, supply_source: str, *, description: str | None = None) -> MeasurementSpec:
    expr = f".measure tran {name} param='-AVG(V({supply_source})*I({supply_source})) * 1e3'"
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
