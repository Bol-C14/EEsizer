# 2026-01-25 2152 Step1 benchmarks

- Added `benchmarks/` with rc/ota/opamp3 benches, relative includes, and DC servo benches for open-loop AC.
- Added per-benchmark `bench.json`/`spec.json` with unified node/supply naming and AC+DC sim plans.
- Extended deck builder to support `output_probes` (e.g., `i(VDD)`) and multi-node outputs.
- Added vout-aware default metrics for AC magnitude and DC output.
- Added a baseline runner example plus unit/integration tests for the new benchmarks.
