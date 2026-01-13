# 25. Provenance and Fingerprints

Provenance is the audit trail that makes runs reproducible and comparable.

## 25.1 ArtifactFingerprint

**Type:** `eesizer_core.contracts.provenance.ArtifactFingerprint`

Fields:
- `sha256: str`

The string MUST be a lowercase hex digest.

## 25.2 Stable hashing

EEsizer uses stable hashing helpers:
- `stable_hash_str(s: str) -> str`
- `stable_hash_bytes(b: bytes) -> str`
- `stable_hash_json(obj: Any) -> str`

Rules:
- Strings are encoded as UTF-8.
- JSON hashing MUST be stable across dictionary ordering; therefore JSON is serialized with `sort_keys=True`.
- Floats are rendered with a deterministic representation.

## 25.3 Provenance

**Type:** `eesizer_core.contracts.provenance.Provenance`

Fields:
- `operator: str`
- `version: str`
- `start_time: float`
- `end_time: float | None`
- `command: str | None`
- `inputs: dict[str, ArtifactFingerprint]`
- `outputs: dict[str, ArtifactFingerprint]`
- `notes: dict[str, Any]`

Helpers:
- `finish()` sets `end_time`.
- `duration_s()` returns elapsed seconds if finished.

### Requirements

Every operator run MUST return a provenance record that includes:
- enough fingerprints to uniquely identify its inputs and outputs
- tool command line (if applicable)
- timestamps

### Minimal provenance set (recommended)

For tool operators (ngspice, formal engines):
- input artifact hashes
- deck/netlist hash
- tool binary/version string if available
- command line and working directory
- return code and output file paths

## 25.4 RunManifest

**Type:** `eesizer_core.contracts.provenance.RunManifest`

Fields:
- `run_id: str`
- `workspace: Path`
- `seed: int | None`
- `tool_versions: dict[str, str]`
- `env: dict[str, str]`
- `notes: dict[str, Any]`

Helpers:
- `save_json(path: Path)` writes a JSON manifest (parents created).

## 25.5 RunResult

**Type:** `eesizer_core.contracts.artifacts.RunResult`

Fields:
- `best_source: CircuitSource | None`
- `best_metrics: MetricsBundle`
- `history: list[dict[str, Any]]` (per-iteration records)
- `stop_reason: StopReason | None`
- `notes: dict[str, Any]`
