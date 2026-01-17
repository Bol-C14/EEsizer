# 2026-01-15 10:30 — Step 8 legacy metrics baseline

## Summary
- Added a legacy metrics baseline adapter that runs legacy simulations/metrics and records Step 7 artifacts.
- Added `baseline_legacy` stop reason for baseline summaries.
- Exported the legacy baseline entrypoint for reuse.

## Files touched
- Code: `src/eesizer_core/baselines/legacy_metrics_adapter.py`, `src/eesizer_core/baselines/__init__.py`, `src/eesizer_core/contracts/enums.py`, `src/eesizer_core/__init__.py`
- Docs: `docs/specs/25_artifact_provenance.md`

## Notes / Rationale
- Legacy baseline enables direct comparison against legacy metrics without modifying legacy code.
- Adapter records a single provenance entry for legacy metrics execution.

## Migration notes
- New stop reason `baseline_legacy` appears in run summaries for legacy baselines.
