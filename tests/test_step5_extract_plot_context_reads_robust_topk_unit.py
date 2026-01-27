import json

from eesizer_core.analysis.plotting.extract import extract_plot_context
from eesizer_core.runtime.recorder import RunRecorder


def test_extract_plot_context_reads_robust_search_files(tmp_path):
    recorder = RunRecorder(tmp_path / "run")
    # Minimal run skeleton: no history.corners.
    recorder.write_json("inputs/spec.json", {"objectives": []})
    recorder.write_json(
        "search/robust_topk.json",
        [
            {
                "iteration": 1,
                "worst_corner_id": "all_low",
                "nominal_metrics": {"ugbw_hz": 1e6, "power_w": 1e-3},
                "worst_metrics": {"ugbw_hz": 8e5, "power_w": 2e-3},
            }
        ],
    )

    ctx = extract_plot_context(recorder.run_dir)
    assert ctx.robust_rows
    row = ctx.robust_rows[0]
    assert row["iteration"] == 1
    assert row["worst_corner_id"] == "all_low"
    assert row["nominal_metrics"]["ugbw_hz"] == 1e6
    assert row["worst_metrics"]["ugbw_hz"] == 8e5

