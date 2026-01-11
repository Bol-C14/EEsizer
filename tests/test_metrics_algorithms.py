from pathlib import Path

import pytest

from eesizer_core.contracts.enums import SimKind
from eesizer_core.contracts.errors import MetricError, ValidationError
from eesizer_core.sim.artifacts import RawSimData
from eesizer_core.metrics import (
    DEFAULT_REGISTRY,
    MetricSpec,
    compute_ac_mag_db_at,
    compute_unity_gain_freq,
    ComputeMetricsOperator,
)
from eesizer_core.metrics.registry import MetricRegistry


def _make_raw(tmp_path):
    ac_path = tmp_path / "ac.csv"
    fixture = Path(__file__).parent / "fixtures" / "ac.csv"
    ac_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = tmp_path / "ngspice.log"
    log_path.write_text("", encoding="utf-8")
    return RawSimData(
        kind=SimKind.ac,
        run_dir=tmp_path,
        outputs={"ac_csv": ac_path},
        log_path=log_path,
        cmdline=[],
        returncode=0,
    )


def test_ac_mag_db_at(tmp_path):
    raw = _make_raw(tmp_path)
    spec = MetricSpec(
        name="ac_mag_db_at_10",
        unit="dB",
        requires_kind=SimKind.ac,
        requires_outputs=("ac_csv",),
        compute_fn=compute_ac_mag_db_at,
        params={"target_hz": 10, "node": "out"},
    )
    value = compute_ac_mag_db_at(raw, spec)
    assert value == pytest.approx(10.0)


def test_unity_gain_freq(tmp_path):
    raw = _make_raw(tmp_path)
    spec = MetricSpec(
        name="ac_unity_gain_freq",
        unit="Hz",
        requires_kind=SimKind.ac,
        requires_outputs=("ac_csv",),
        compute_fn=compute_unity_gain_freq,
        params={"node": "out"},
    )
    value = compute_unity_gain_freq(raw, spec)
    assert value == pytest.approx(100.0)


def test_compute_metrics_operator_with_registry(tmp_path):
    raw = _make_raw(tmp_path)
    registry = MetricRegistry(
        {
            "ac_mag_db_at_1k": MetricSpec(
                name="ac_mag_db_at_1k",
                unit="dB",
                requires_kind=SimKind.ac,
                requires_outputs=("ac_csv",),
                compute_fn=compute_ac_mag_db_at,
                params={"target_hz": 1000, "node": "out"},
            ),
        }
    )

    op = ComputeMetricsOperator(registry=registry)
    result = op.run({"raw_data": raw, "metric_names": ["ac_mag_db_at_1k"]}, ctx=None)

    metrics = result.outputs["metrics"]
    assert metrics.values["ac_mag_db_at_1k"].value == pytest.approx(-10.0)


def test_unknown_metric_raises(tmp_path):
    raw = _make_raw(tmp_path)
    op = ComputeMetricsOperator(registry=DEFAULT_REGISTRY)
    with pytest.raises(ValidationError):
        op.run({"raw_data": raw, "metric_names": ["missing_metric"]}, ctx=None)
