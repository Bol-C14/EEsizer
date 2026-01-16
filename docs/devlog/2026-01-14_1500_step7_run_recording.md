# 2026-01-14 15:00 — Step 7 run recording

## Summary
- Added `RunRecorder` and `RunLoader` to persist run artifacts (inputs/history/provenance/best) and reload them.
- Expanded `RunManifest` with timestamps, input hashes, file index, and result summary.
- Integrated recorder/manifest into `PatchLoopStrategy` and added Step 7 tests.

## Files touched
- Code: `src/eesizer_core/runtime/recorder.py`, `src/eesizer_core/runtime/run_loader.py`, `src/eesizer_core/runtime/context.py`, `src/eesizer_core/contracts/provenance.py`, `src/eesizer_core/strategies/patch_loop.py`
- Docs: `docs/specs/11_runtime_layout.md`, `docs/specs/25_artifact_provenance.md`, `docs/templates/run_manifest.example.json`, `docs/specs/schemas/run_manifest.example.json`, `docs/wiki/09_troubleshooting.md`
- Tests: `tests/test_step7_recorder_smoke.py`, `tests/test_step7_manifest_hashes.py`, `tests/test_step7_run_outputs_mock.py`, `tests/test_step7_run_loader.py`

## Notes / Rationale
- Run outputs are now deterministic and portable via relative paths and JSONL logs.
- Manifest captures hashes and summary metadata for reproducibility audits.
