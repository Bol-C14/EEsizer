"""Measurement helpers and aggregation utilities shared across agents."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

from ..spice import MeasurementSpec, measure_gain, measure_power, measure_thd


def output_swing_measurements(output_node: str, *, prefix: str = "output_swing") -> Sequence[MeasurementSpec]:
    """Measurements that capture max/min/peak-to-peak output swing in transient."""

    return (
        MeasurementSpec(
            name=f"{prefix}_max",
            statement=f".measure tran {prefix}_max max V({output_node})",
            analysis="tran",
            description="Maximum output voltage during the transient window",
        ),
        MeasurementSpec(
            name=f"{prefix}_min",
            statement=f".measure tran {prefix}_min min V({output_node})",
            analysis="tran",
            description="Minimum output voltage during the transient window",
        ),
        MeasurementSpec(
            name=f"{prefix}_pp",
            statement=f".measure tran {prefix}_pp param='({prefix}_max-{prefix}_min)'",
            analysis="tran",
            description="Peak-to-peak swing computed from extrema",
        ),
    )


def offset_measurement(output_node: str, *, name: str = "offset_v") -> MeasurementSpec:
    """Input-referred offset captured via DC operating point."""

    return MeasurementSpec(
        name=name,
        statement=f".measure dc {name} param='V({output_node})'",
        analysis="dc",
        description="DC output voltage treated as input-referred offset",
    )


def icmr_measurements(input_node: str, *, prefix: str = "icmr") -> Sequence[MeasurementSpec]:
    """Common-mode input range derived from the DC sweep of the input node."""

    return (
        MeasurementSpec(
            name=f"{prefix}_min_v",
            statement=f".measure dc {prefix}_min_v min V({input_node})",
            analysis="dc",
            description="Minimum input common-mode voltage encountered in sweep",
        ),
        MeasurementSpec(
            name=f"{prefix}_max_v",
            statement=f".measure dc {prefix}_max_v max V({input_node})",
            analysis="dc",
            description="Maximum input common-mode voltage encountered in sweep",
        ),
    )


def cmrr_measurement(output_node: str, input_node: str, *, name: str = "cmrr_db") -> MeasurementSpec:
    """Common-mode rejection ratio based on output/input AC response."""

    return MeasurementSpec(
        name=name,
        statement=(
            f".measure ac {name} param='20*log10(abs(V({output_node})/V({input_node})))'"
        ),
        analysis="ac",
        description="Common-mode rejection ratio derived from AC sweep",
    )


def bandwidth_measurements(output_node: str, *, prefix: str = "bandwidth") -> Sequence[MeasurementSpec]:
    """-3dB and unity-gain bandwidth approximations based on AC magnitude."""

    return (
        MeasurementSpec(
            name=f"{prefix}_hz",
            statement=f".measure ac {prefix}_hz when vdb({output_node})=-3",
            analysis="ac",
            description="Approximate -3dB bandwidth from AC response",
        ),
        MeasurementSpec(
            name="unity_bandwidth_hz",
            statement=".measure ac unity_bandwidth_hz when vdb({})=0".format(output_node),
            analysis="ac",
            description="Unity-gain bandwidth inferred from AC magnitude",
        ),
    )


def gain_measurements(output_node: str, input_node: str) -> Sequence[MeasurementSpec]:
    """AC and transient gain helpers following notebook conventions."""

    return (
        measure_gain(
            "ac_gain_db",
            output_node=output_node,
            input_node=input_node,
            analysis="ac",
            description="Small-signal AC gain",
        ),
        MeasurementSpec(
            name="tran_gain_db",
            statement=(
                ".measure tran tran_gain_db param='20*log10(abs(V({})/V({})))'".format(
                    output_node, input_node
                )
            ),
            analysis="tran",
            description="Large-signal transient gain",
        ),
    )


def thd_measurement(output_node: str, fundamental_hz: float) -> MeasurementSpec:
    """THD measurement around the chosen transient fundamental."""

    return measure_thd(
        "thd_output_db",
        node=output_node,
        fundamental_hz=fundamental_hz,
        description="Total harmonic distortion of the output node",
    )


def standard_measurements(
    output_node: str,
    input_node: str,
    supply_source: str,
    *,
    fundamental_hz: float = 1e3,
) -> Sequence[MeasurementSpec]:
    """Aggregate measurement suite mirroring the notebook tool-calling helpers."""

    measurements: list[MeasurementSpec] = []
    measurements.extend(gain_measurements(output_node, input_node))
    measurements.extend(bandwidth_measurements(output_node))
    measurements.append(thd_measurement(output_node, fundamental_hz))
    measurements.extend(output_swing_measurements(output_node))
    measurements.append(offset_measurement(output_node))
    measurements.extend(icmr_measurements(input_node))
    measurements.append(cmrr_measurement(output_node, input_node))
    measurements.append(
        measure_power(
            "power_mw",
            supply_source=supply_source,
            description="Average power consumption converted to mW",
        )
    )
    return tuple(measurements)


def merge_metric_sources(*sources: Mapping[str, float]) -> MutableMapping[str, float]:
    """Merge metric dictionaries while preserving float typing and latest values."""

    merged: MutableMapping[str, float] = {}
    for source in sources:
        for key, value in source.items():
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
    return merged


def aggregate_measurement_values(raw: Mapping[str, float]) -> MutableMapping[str, float]:
    """Normalize raw measurement names and derive composed metrics."""

    metrics: MutableMapping[str, float] = merge_metric_sources(raw)
    if "ac_gain_db" in raw:
        metrics.setdefault("gain_db", float(raw["ac_gain_db"]))
    if "tran_gain_db" in raw:
        metrics.setdefault("tran_gain_db", float(raw["tran_gain_db"]))
    if "output_swing_max" in raw and "output_swing_min" in raw:
        metrics.setdefault(
            "output_swing_v", float(raw["output_swing_max"]) - float(raw["output_swing_min"])
        )
    if "output_swing_pp" in raw:
        metrics.setdefault("output_swing_v", float(raw["output_swing_pp"]))
    if "offset_v" in raw:
        metrics.setdefault("offset_v", float(raw["offset_v"]))
    if "vout_dc" in raw and "offset_v" not in metrics:
        metrics["offset_v"] = float(raw["vout_dc"])
    if "thd_output_db" in raw:
        metrics.setdefault("thd_db", float(raw["thd_output_db"]))
    if "cmrr_db" in raw:
        metrics.setdefault("cmrr_db", float(raw["cmrr_db"]))
    if "bandwidth_hz" in raw:
        metrics.setdefault("bandwidth_hz", float(raw["bandwidth_hz"]))
    if "unity_bandwidth_hz" in raw:
        metrics.setdefault("unity_bandwidth_hz", float(raw["unity_bandwidth_hz"]))
    if "icmr_min_v" in raw:
        metrics.setdefault("icmr_min_v", float(raw["icmr_min_v"]))
    if "icmr_max_v" in raw:
        metrics.setdefault("icmr_max_v", float(raw["icmr_max_v"]))
    return metrics


def validate_metrics(
    metrics: Mapping[str, float], *, required: Sequence[str] = ("gain_db", "power_mw")
) -> Sequence[str]:
    """Return the list of required metrics that are missing from the payload."""

    missing = [name for name in required if name not in metrics]
    return tuple(missing)


__all__ = [
    "merge_metric_sources",
    "aggregate_measurement_values",
    "bandwidth_measurements",
    "cmrr_measurement",
    "gain_measurements",
    "validate_metrics",
    "icmr_measurements",
    "offset_measurement",
    "output_swing_measurements",
    "standard_measurements",
    "thd_measurement",
]
