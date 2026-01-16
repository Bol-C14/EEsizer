import json

from eesizer_core.runtime.recorder import RunRecorder


def test_run_recorder_writes_inputs_and_jsonl(tmp_path):
    recorder = RunRecorder(tmp_path)
    recorder.write_input("source.sp", "* test\n.end\n")
    recorder.write_json("inputs/spec.json", {"name": "demo"})
    recorder.append_jsonl("history/iterations.jsonl", {"iter": 0})
    recorder.append_jsonl("history/iterations.jsonl", {"iter": 1})

    assert (tmp_path / "inputs" / "source.sp").exists()
    assert (tmp_path / "inputs" / "spec.json").exists()

    lines = (tmp_path / "history" / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["iter"] == 0
