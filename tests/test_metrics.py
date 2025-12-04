from eesizer_core.analysis.metrics import (
    aggregate_measurement_values,
    merge_metric_sources,
    validate_metrics,
)


def test_merge_metric_sources_prefers_latest():
    merged = merge_metric_sources({"gain_db": 10.0}, {"gain_db": 20, "power_mw": "3.1"})
    assert merged["gain_db"] == 20.0
    assert merged["power_mw"] == 3.1


def test_validate_metrics_flags_missing():
    missing = validate_metrics({"gain_db": 10.0}, required=("gain_db", "power_mw"))
    assert missing == ("power_mw",)


def test_aggregate_measurement_values_derives_gain_and_swing():
    metrics = aggregate_measurement_values(
        {"ac_gain_db": 40, "output_swing_max": 1.2, "output_swing_min": 0.2}
    )
    assert metrics["gain_db"] == 40.0
    assert metrics["output_swing_v"] == 1.0


def test_aggregate_measurement_values_derives_power_margin_and_icmr_range():
    metrics = aggregate_measurement_values(
        {
            "power_w": 0.002,
            "tran_gain_db": 55,
            "icmr_min_v": 0.1,
            "icmr_max_v": 1.1,
            "phase_margin_deg": 72.0,
        }
    )
    assert metrics["power_mw"] == 2.0
    assert metrics["gain_db"] == 55.0
    assert metrics["icmr_range_v"] == 1.0
    assert metrics["phase_margin_deg"] == 72.0
