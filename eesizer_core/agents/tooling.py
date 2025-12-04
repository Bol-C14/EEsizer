"""Reusable helpers for building simulation control decks."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..analysis.metrics import standard_measurements
from ..spice import (
    ControlDeck,
    SimulationDirective,
    WaveformExport,
    ac_simulation,
    dc_simulation,
    tran_simulation,
)


def guess_identifier(netlist_text: str | None, candidates: Sequence[str]) -> str | None:
    """Return the first candidate that appears in the netlist text."""

    if not isinstance(netlist_text, str):
        return None
    for candidate in candidates:
        if not candidate:
            continue
        pattern = re.compile(rf"\b{re.escape(candidate)}\b", re.IGNORECASE)
        match = pattern.search(netlist_text)
        if match:
            return match.group(0)
    return None


def resolve_identifier(
    provided: object,
    netlist_text: str | None,
    fallbacks: Sequence[str],
    *,
    default: str,
) -> str:
    """Pick an identifier using user input, netlist hints, then fallbacks."""

    candidates: list[str] = []
    if isinstance(provided, str) and provided.strip():
        candidates.append(provided.strip())
    candidates.extend(fallbacks)
    guess = guess_identifier(netlist_text, tuple(candidates))
    if guess:
        return guess
    if candidates:
        return candidates[0]
    return default


def build_default_deck(
    arguments: Mapping[str, Any],
    *,
    netlist_text: str | None = None,
) -> ControlDeck:
    """Assemble a control deck (DC/AC/TRAN) plus waveform exports and measurements."""

    deck_name = str(arguments.get("deck_name", "ota_ac_combo"))
    output_node = resolve_identifier(
        arguments.get("output_node"),
        netlist_text,
        ("out", "vout", "outp", "vout+"),
        default="out",
    )
    input_node = resolve_identifier(
        arguments.get("input_node"),
        netlist_text,
        ("diffin", "vin", "inp", "in", "vinp", "vip", "in1"),
        default="in",
    )
    supply_source = resolve_identifier(
        arguments.get("supply_source"),
        netlist_text,
        ("vdd", "vcc", "vdd!"),
        default="vdd",
    )
    dc_source = resolve_identifier(
        arguments.get("dc_source"),
        netlist_text,
        ("vid", "vin", "vinp"),
        default="VIN",
    )
    directives: tuple[SimulationDirective, ...] = (
        dc_simulation(
            source=dc_source,
            start=float(arguments.get("dc_start", -0.1)),
            stop=float(arguments.get("dc_stop", 0.1)),
            step=float(arguments.get("dc_step", 0.01)),
            description="DC operating point sweep",
        ),
        ac_simulation(
            sweep=str(arguments.get("ac_sweep", "dec")),
            points=int(arguments.get("ac_points", 40)),
            start_hz=float(arguments.get("ac_start", 10.0)),
            stop_hz=float(arguments.get("ac_stop", 1e6)),
            description="AC magnitude sweep",
        ),
        tran_simulation(
            step=float(arguments.get("tran_step", 1e-7)),
            stop=float(arguments.get("tran_stop", 1e-3)),
            description="Transient window for THD/power",
        ),
    )
    save_targets = {
        f"V({output_node})",
        f"V({input_node})",
        f"V({supply_source})",
        f"I({supply_source})",
    }
    waveform_exports: tuple[WaveformExport, ...] = (
        WaveformExport(
            filename="output_tran.dat",
            vectors=(
                "time",
                f"V({output_node})",
                f"V({input_node})",
                f"V({supply_source})",
                f"I({supply_source})",
            ),
            plot="tran1",
            description="Transient waveforms for THD/power derivation",
        ),
        WaveformExport(
            filename="output_ac.dat",
            vectors=(
                "frequency",
                f"vdb({output_node})",
                f"vp({output_node})",
            ),
            plot="ac1",
            description="AC magnitude and phase for gain/bandwidth/phase margin derivation",
        ),
    )
    measurements = standard_measurements(
        output_node=output_node,
        input_node=input_node,
        supply_source=supply_source,
        fundamental_hz=float(arguments.get("tran_fundamental", 1e3)),
    )
    return ControlDeck(
        name=deck_name,
        directives=directives,
        measurements=measurements,
        saves=tuple(sorted(save_targets)),
        waveform_exports=waveform_exports,
    )


__all__ = ["build_default_deck", "resolve_identifier", "guess_identifier"]
