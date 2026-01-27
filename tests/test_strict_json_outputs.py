import json

from eesizer_core.contracts.provenance import RunManifest
from eesizer_core.runtime.recorder import RunRecorder


def test_run_recorder_sanitizes_non_finite_numbers(tmp_path) -> None:
    recorder = RunRecorder(tmp_path / "run")

    out = recorder.write_json("example.json", {"a": float("inf"), "b": float("nan"), "c": 1.0})
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["a"] is None
    assert loaded["b"] is None
    assert loaded["c"] == 1.0

    out2 = recorder.append_jsonl("example.jsonl", {"x": float("-inf")})
    line = out2.read_text(encoding="utf-8").splitlines()[-1]
    assert "Infinity" not in line
    assert "NaN" not in line
    loaded_line = json.loads(line)
    assert loaded_line["x"] is None


def test_run_manifest_save_json_sanitizes_non_finite_numbers(tmp_path) -> None:
    manifest = RunManifest(run_id="run-1", workspace=tmp_path)
    manifest.result_summary["best_score"] = float("inf")
    path = tmp_path / "run_manifest.json"
    manifest.save_json(path)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["result_summary"]["best_score"] is None

