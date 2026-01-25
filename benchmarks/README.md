# Benchmarks

This folder contains reproducible benchmark testbenches used by the baseline and search strategies.

Structure:
- models/: shared device models (relative .include only)
- rc/, ota/, opamp3/: per-benchmark netlists + metadata
- suites/: optional suite definitions

Conventions:
- Output node: vout
- Inputs: vin (rc), vinp/vinn (ota/opamp3)
- Supply source name: VDD (node vdd), optional VSS
- .include paths are repo-relative (no absolute paths, no ..)
