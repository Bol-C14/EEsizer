from pathlib import Path

from eesizer_core.analysis.compare_runs import compare_runs, _objective_table
from eesizer_core.runtime.recorder import RunRecorder


def _write_run(run_dir: Path, run_id: str, metrics_payload: dict) -> None:
    recorder = RunRecorder(run_dir)
    spec_payload = {
        "objectives": [
            {"metric": "gain_db", "target": 9.0, "tol": None, "weight": 1.0, "sense": "ge"},
        ],
        "constraints": [],
        "observables": [],
        "notes": {},
    }
    summary_payload = {
        "stop_reason": "baseline_noopt",
        "best_iter": 0,
        "best_score": 0.0,
        "sim_runs_total": 1,
    }
    manifest_payload = {
        "run_id": run_id,
        "timestamp_start": "2026-01-01T00:00:00+00:00",
        "timestamp_end": "2026-01-01T00:00:01+00:00",
        "result_summary": summary_payload,
    }

    recorder.write_input("spec.json", spec_payload)
    recorder.write_json("best/best_metrics.json", metrics_payload)
    recorder.write_json("history/summary.json", summary_payload)
    recorder.append_jsonl("history/iterations.jsonl", {"iteration": 0, "attempts": [{}]})
    recorder.write_json("run_manifest.json", manifest_payload)


def test_compare_runs_outputs(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_run(
        run_a,
        "run_a",
        {"gain_db": {"value": 10.0, "unit": "dB"}},
    )
    _write_run(
        run_b,
        "run_b",
        {"gain_db": {"value": 10.1, "unit": "dB"}},
    )

    out_dir = tmp_path / "comparison"
    comparison = compare_runs(run_a, run_b, out_dir)

    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "report.md").exists()
    assert comparison["metrics"]["gain_db"]["within_tol"] is True

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Metrics Diff" in report


def test_objective_table_handles_mismatch():
    rows_a = [{"metric": "gain_db", "passed": True}]
    rows_b = [{"metric": "gain_db", "passed": False}, {"metric": "pm_deg", "passed": True}]
    table = _objective_table(rows_a, rows_b)
    assert {row["metric"] for row in table} == {"gain_db", "pm_deg"}
    lookup = {row["metric"]: row for row in table}
    assert lookup["gain_db"]["a"] is True
    assert lookup["gain_db"]["b"] is False
    assert lookup["pm_deg"]["a"] is None
    assert lookup["pm_deg"]["b"] is True
