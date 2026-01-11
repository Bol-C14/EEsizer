# DevLog: Step4 wrdata Loader Hardening

- Date: 2026-01-11 18:30 (UTC)
- Scope: Harden wrdata parsing and align metrics with ngspice output quirks.

## Problem
ngspice `wrdata` files often:
- include a header line prefixed with `*`, or
- have no header at all (first line is data).

Previous metrics used `pandas.read_csv(sep=r"\s+")` with assumed headers, risking silent mis-reads or errors.

## Changes
- Added `metrics/io.py::load_wrdata_table`:
  - Detects comment headers, infers header from non-numeric first line, or applies `expected_columns` when no header.
  - Skips comment lines in data, checks column counts, and raises `MetricError` on mismatch.
- Updated AC/DC/TRAN metrics to use the loader with expected column definitions from deck templates.
- Clarified fixtures to cover common formats:
  - AC: comment-prefixed header.
  - DC: no header (numeric only).
  - TRAN: standard header.

## Notes
- Files are whitespace-separated despite `.csv` suffix; loader handles both headered and headerless tables.
- Adding new metrics should continue to pass expected column lists that match deck `wrdata` definitions to avoid silent drift.
