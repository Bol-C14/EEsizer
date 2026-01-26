from eesizer_core.metrics.aliases import canonicalize_metrics


def test_canonicalize_metrics_aliases():
    metrics = {
        "gain_output": {"value": 12.0, "unit": "dB"},
        "pm": {"value": 60.0, "unit": "deg"},
        "bw": 1e6,
        "ubw_output": {"value": 2e6, "unit": "Hz"},
        "out_swing_v": {"value": 0.5, "unit": "V"},
    }

    result = canonicalize_metrics(metrics)

    assert "gain_db" in result
    assert "phase_margin_deg" in result
    assert "bw_3db_hz" in result
    assert "ugbw_hz" in result
    assert "out_swing_v" in result
    assert "gain_output" not in result
    assert "pm" not in result
    assert "bw" not in result

    assert result["gain_db"]["value"] == 12.0
    assert result["gain_db"]["unit"] == "dB"
    assert result["phase_margin_deg"]["value"] == 60.0
    assert result["bw_3db_hz"] == 1e6
    assert result["ugbw_hz"]["value"] == 2e6
