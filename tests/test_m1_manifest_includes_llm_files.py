from pathlib import Path

from eesizer_core.contracts import MetricsBundle
from eesizer_core.contracts.enums import StopReason
from eesizer_core.contracts.provenance import RunManifest
from eesizer_core.runtime.recorder import RunRecorder
from eesizer_core.strategies.patch_loop import _finalize_run


def test_manifest_includes_llm_files(tmp_path):
    recorder = RunRecorder(tmp_path)
    llm_path = Path(tmp_path) / "llm" / "llm_i001_a00" / "prompt.txt"
    llm_path.parent.mkdir(parents=True, exist_ok=True)
    llm_path.write_text("prompt", encoding="utf-8")

    manifest = RunManifest(run_id="test", workspace=tmp_path)
    _finalize_run(
        recorder=recorder,
        manifest=manifest,
        best_source=None,
        best_metrics=MetricsBundle(),
        history=[],
        stop_reason=StopReason.policy_stop,
        best_score=1.0,
        best_iter=None,
        sim_runs=0,
        sim_runs_ok=0,
        sim_runs_failed=0,
    )

    llm_files = manifest.files.get("llm", [])
    assert "llm/llm_i001_a00/prompt.txt" in llm_files
