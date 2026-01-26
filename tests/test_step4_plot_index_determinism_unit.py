import json

from eesizer_core.analysis.plotting.plot_index import PlotEntry, build_plot_index


def test_plot_index_determinism():
    entries = [
        PlotEntry(
            name="plot_b",
            png_path="plots/plot_b.png",
            data_path="plots/plot_b_data.json",
            data_sha256="b",
            status="ok",
        ),
        PlotEntry(
            name="plot_a",
            png_path=None,
            data_path="plots/plot_a_data.json",
            data_sha256="a",
            status="skipped",
            skip_reason="no_data",
        ),
    ]

    index_a = build_plot_index(entries)
    index_b = build_plot_index(entries)

    assert json.dumps(index_a, sort_keys=True) == json.dumps(index_b, sort_keys=True)
    assert index_a["plots"][0]["name"] == "plot_a"
