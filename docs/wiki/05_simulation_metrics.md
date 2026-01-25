# Simulation & Metrics

This document standardises how we run simulators and compute metrics.

## 1) SimPlan
SimPlan is the “what to run” spec:
- sim types: `dc`, `ac`, `tran` (SimKind.ams is reserved for future)
- sweeps: frequency range, time step, etc.
- outputs: which traces to save, which files to produce (node probes plus optional raw expressions like `i(VDD)`)
- corners: optional list of PVT corners

## 2) Operators (recommended split)
1) `DeckBuildOperator`: `CircuitIR/CircuitSource + SimPlan -> SpiceDeck`
2) `NgspiceRunOperator`: `SpiceDeck -> RawSimData`
3) `ComputeMetricsOperator`: `RawSimData -> MetricsBundle`

Do not merge these into a single mega-file.

### Deck/Runner rules (Step4)
- `SpiceDeck.expected_outputs` names are always written to the run directory (`<workspace_root>/runs/<run_id>/<stage>/...`).
- `SpiceDeck.expected_outputs_meta` records the wrdata column order (e.g., `("frequency", "real(v(out))", "imag(v(out))")`). For multiple outputs, the scale column appears once, followed by each probe expression in order.
- `SpiceDeck.workdir` is mandatory when the source netlist uses relative `.include`/`.lib`; `NgspiceRunOperator` runs with `cwd=workdir` and rewrites wrdata targets to absolute paths under the run directory.

## 3) Metric registry
Every metric must have:
- name (stable id)
- units
- required sim type
- required outputs (files/traces)
- implementation reference

This prevents “same metric name, different definitions”.

### Naming (contract vs implementation)
- Contract layer: `contracts.artifacts.MetricSpec` = declaration of required sim/output/unit.
- Implementation layer: `metrics.registry.MetricImplSpec` = executable definition with `compute_fn`.
- Mapping: Strategy/Policy should depend on the contract spec; operators/registry use the impl spec.

## 4) Adding a new metric
1) Add metric spec to registry
2) Implement calculator under `eesizer_core/metrics/`
3) Add unit tests (synthetic inputs preferred)
4) Add to example spec if relevant

## 5) Provenance requirements
Simulator operator must record:
- ngspice binary path + version
- command-line used
- stdout/stderr + exit status
- hashes of deck and netlist

Metric operator must record:
- which files were read
- any fallback behavior (e.g., missing columns)

## 6) wrdata file format (ngspice)
- Files are whitespace-separated tables (often named `.csv` but not comma-separated).
- Header handling:
  - If the first non-empty line starts with `*`, treat it as header (sans `*`).
  - If the first line is non-numeric, treat it as header.
- If no header, use the column order from `SpiceDeck.expected_outputs_meta`.
- If `expected_columns` is provided, it overrides header names while preserving column order.
- `load_wrdata_table` (in `eesizer_core/io/ngspice_wrdata.py`) is the single loader used by all metrics; do not reimplement parsing.
- Metrics must raise clear errors when required columns are missing (no silent fallbacks).

## 7) Known legacy issues to avoid
See `legacy/docs/code_review_2025-12-31.md` for prior metric correctness notes.
