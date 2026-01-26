from eesizer_core.analysis.plotting.plot_data import build_tradeoff_data


def test_tradeoff_data_marks_missing():
    rows = [
        {
            "iteration": 1,
            "metrics": {"power_w": 1e-3, "ugbw_hz": 1e6},
            "status": "ok",
            "tags": [],
        },
        {
            "iteration": 2,
            "metrics": {"power_w": None, "ugbw_hz": 2e6},
            "status": "metric_missing",
            "tags": [],
        },
    ]
    data = build_tradeoff_data(rows, x_metric="power_w", y_metric="ugbw_hz")
    points = data["points"]

    assert points[0]["x"] == 1e-3
    assert points[0]["y"] == 1e6
    assert points[1]["x"] is None
    assert points[1]["status"] == "metric_missing"
