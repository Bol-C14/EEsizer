from pathlib import Path

import pytest

from eesizer_core.runtime.recorder import RunRecorder


def test_recorder_rejects_absolute_paths(tmp_path):
    recorder = RunRecorder(tmp_path)
    abs_path = tmp_path / "evil.txt"

    with pytest.raises(ValueError, match="relative"):
        recorder.write_text(str(abs_path), "nope")


def test_recorder_rejects_parent_traversal(tmp_path):
    recorder = RunRecorder(tmp_path)

    with pytest.raises(ValueError, match="\\.\\."):
        recorder.write_text("../evil.txt", "nope")


def test_recorder_writes_within_run_dir(tmp_path):
    recorder = RunRecorder(tmp_path)
    out_path = recorder.write_json("safe/output.json", {"ok": True})

    assert out_path == (Path(tmp_path) / "safe" / "output.json").resolve()
    assert out_path.read_text(encoding="utf-8")
