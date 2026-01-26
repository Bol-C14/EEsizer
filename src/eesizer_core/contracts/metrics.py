from __future__ import annotations

from dataclasses import dataclass


UGBW_HZ = "ugbw_hz"
PHASE_MARGIN_DEG = "phase_margin_deg"
POWER_W = "power_w"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    definition: str
    requirements: str = ""


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    UGBW_HZ: MetricDefinition(
        name=UGBW_HZ,
        unit="Hz",
        definition=(
            "First 0 dB crossing of |A(jw)| with A=V(vout)/(V(vinp)-V(vinn)); "
            "vinp ac=1, vinn ac=0."
        ),
        requirements="AC probes: v(vout), v(vinp), v(vinn).",
    ),
    PHASE_MARGIN_DEG: MetricDefinition(
        name=PHASE_MARGIN_DEG,
        unit="deg",
        definition="PM = 180 + phase(A) at UGBW; phase is unwrapped.",
        requirements="AC probes: v(vout), v(vinp), v(vinn).",
    ),
    POWER_W: MetricDefinition(
        name=POWER_W,
        unit="W",
        definition="Power = abs(I(VDD)) * VDD from DC operating point.",
        requirements="DC probes: i(VDD), v(vdd).",
    ),
}
