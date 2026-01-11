# Simulation & Metrics

This document standardises how we run simulators and compute metrics.

## 1) SimPlan
SimPlan is the “what to run” spec:
- sim types: `dc`, `ac`, `tran`, `op`
- sweeps: frequency range, time step, etc.
- outputs: which traces to save, which files to produce
- corners: optional list of PVT corners

## 2) Operators (recommended split)
1) `DeckBuildOperator`: `CircuitIR + SimPlan -> SpiceDeck`
2) `NgspiceRunOperator`: `SpiceDeck -> RawSimData`
3) `MetricOperator`: `RawSimData -> MetricValue`

Do not merge these into a single mega-file.

## 3) Metric registry
Every metric must have:
- name (stable id)
- units
- required sim type
- required outputs (files/traces)
- implementation reference

This prevents “same metric name, different definitions”.

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

## 6) Known legacy issues to avoid
See `legacy/docs/code_review_2025-12-31.md` for prior metric correctness notes.
