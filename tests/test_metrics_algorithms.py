from pathlib import Path

import pytest

from eesizer_core.contracts.enums import SimKind
from eesizer_core.contracts.errors import ValidationError
from eesizer_core.sim.artifacts import RawSimData
from eesizer_core.metrics import (
    DEFAULT_REGISTRY,
    MetricImplSpec,
    compute_ac_mag_db_at,
    compute_unity_gain_freq,
    compute_dc_vout_last,
    compute_dc_slope,
    compute_tran_rise_time,
    ComputeMetricsOperator,
)
from eesizer_core.metrics.registry import MetricRegistry


def _make_raw(tmp_path):
    ac_path = tmp_path / "ac.csv"
    dc_path = tmp_path / "dc.csv"
    tran_path = tmp_path / "tran.csv"
    fixtures_dir = Path(__file__).parent / "fixtures"
    ac_path.write_text((fixtures_dir / "ac.csv").read_text(encoding="utf-8"), encoding="utf-8")
    dc_path.write_text((fixtures_dir / "dc.csv").read_text(encoding="utf-8"), encoding="utf-8")
    tran_path.write_text((fixtures_dir / "tran.csv").read_text(encoding="utf-8"), encoding="utf-8")

    log_path = tmp_path / "ngspice.log"
    log_path.write_text("", encoding="utf-8")
    return {
        "ac": RawSimData(
            kind=SimKind.ac,
            run_dir=tmp_path,
            outputs={"ac_csv": ac_path},
            outputs_meta={"ac_csv": ("frequency", "real(v(out))", "imag(v(out))")},
            log_path=log_path,
            cmdline=[],
            returncode=0,
        ),
        "dc": RawSimData(
            kind=SimKind.dc,
            run_dir=tmp_path,
            outputs={"dc_csv": dc_path},
            outputs_meta={"dc_csv": ("v(in)", "v(out)")},
            log_path=log_path,
            cmdline=[],
            returncode=0,
        ),
        "tran": RawSimData(
            kind=SimKind.tran,
            run_dir=tmp_path,
            outputs={"tran_csv": tran_path},
            outputs_meta={"tran_csv": ("time", "v(out)")},
            log_path=log_path,
            cmdline=[],
            returncode=0,
        ),
    }


def test_ac_mag_db_at(tmp_path):
    raw = _make_raw(tmp_path)["ac"]
    spec = MetricImplSpec(
        name="ac_mag_db_at_10",
        unit="dB",
        requires_kind=SimKind.ac,
        requires_outputs=("ac_csv",),
        compute_fn=compute_ac_mag_db_at,
        params={"target_hz": 10, "node": "out"},
    )
    value, _ = compute_ac_mag_db_at(raw, spec)
    assert value == pytest.approx(0.0)


def test_unity_gain_freq(tmp_path):
    raw = _make_raw(tmp_path)["ac"]
    spec = MetricImplSpec(
        name="ac_unity_gain_freq",
        unit="Hz",
        requires_kind=SimKind.ac,
        requires_outputs=("ac_csv",),
        compute_fn=compute_unity_gain_freq,
        params={"node": "out"},
    )
    value, diag = compute_unity_gain_freq(raw, spec)
    assert diag == {}
    assert value == pytest.approx(10.0)


def test_compute_metrics_operator_with_registry(tmp_path):
    raw = _make_raw(tmp_path)["ac"]
    registry = MetricRegistry(
        {
            "ac_mag_db_at_1k": MetricImplSpec(
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
    assert metrics.values["ac_mag_db_at_1k"].value == pytest.approx(-40.0)


def test_unknown_metric_raises(tmp_path):
    raw = _make_raw(tmp_path)["ac"]
    op = ComputeMetricsOperator(registry=DEFAULT_REGISTRY)
    with pytest.raises(ValidationError):
        op.run({"raw_data": raw, "metric_names": ["missing_metric"]}, ctx=None)


def test_missing_required_output_raises(tmp_path):
    log_path = tmp_path / "ngspice.log"
    log_path.write_text("", encoding="utf-8")
    raw = RawSimData(
        kind=SimKind.ac,
        run_dir=tmp_path,
        outputs={},
        outputs_meta={},
        log_path=log_path,
        cmdline=[],
        returncode=0,
    )
    op = ComputeMetricsOperator(registry=DEFAULT_REGISTRY)
    result = op.run({"raw_data": raw, "metric_names": ["ac_mag_db_at_1k"]}, ctx=None)
    metrics = result.outputs["metrics"]
    assert metrics.values["ac_mag_db_at_1k"].value is None
    assert metrics.values["ac_mag_db_at_1k"].details["status"] == "missing"


def test_dc_metrics(tmp_path):
    raw = _make_raw(tmp_path)["dc"]
    spec_last = MetricImplSpec(
        name="dc_vout_last",
        unit="V",
        requires_kind=SimKind.dc,
        requires_outputs=("dc_csv",),
        compute_fn=compute_dc_vout_last,
        params={"node": "out"},
    )
    value, _ = compute_dc_vout_last(raw, spec_last)
    assert value == pytest.approx(0.5)

    spec_slope = MetricImplSpec(
        name="dc_slope",
        unit="V/V",
        requires_kind=SimKind.dc,
        requires_outputs=("dc_csv",),
        compute_fn=compute_dc_slope,
        params={"sweep_col": "v(in)", "node": "out"},
    )
    value, diag = compute_dc_slope(raw, spec_slope)
    assert diag == {}
    assert value == pytest.approx(0.5)


def test_tran_rise_time(tmp_path):
    raw = _make_raw(tmp_path)["tran"]
    spec = MetricImplSpec(
        name="tran_rise_time",
        unit="s",
        requires_kind=SimKind.tran,
        requires_outputs=("tran_csv",),
        compute_fn=compute_tran_rise_time,
        params={"node": "out"},
    )
    value, diag = compute_tran_rise_time(raw, spec)
    assert diag == {} or diag is None
    assert value == pytest.approx(0.0002)
