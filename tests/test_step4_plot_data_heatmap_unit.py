from eesizer_core.analysis.plotting.plot_data import build_heatmap_data


def test_heatmap_data_shape_and_labels():
    rows = [
        {
            "iteration": 1,
            "candidate": {"p1": 1.0, "p2": 2.0},
            "deltas": {"p1": {"log10": 0.0}, "p2": {"log10": 0.3}},
            "tags": ["topk"],
        },
        {
            "iteration": 2,
            "candidate": {"p1": 1.5, "p2": 1.0},
            "deltas": {"p1": {"log10": 0.1761}, "p2": {"log10": -0.3}},
            "tags": ["pareto"],
        },
    ]
    data = build_heatmap_data(rows, ["p1", "p2"])

    assert data["plot"] == "knob_delta_heatmap"
    assert len(data["matrix"]) == 2
    assert len(data["matrix"][0]) == 2
    assert data["row_labels"][0].startswith("i1")
    assert "T" in data["row_labels"][0]
