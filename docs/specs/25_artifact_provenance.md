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
- `command: str | None`
- `inputs: dict[str, ArtifactFingerprint]`
- `outputs: dict[str, ArtifactFingerprint]`
- `meta: dict[str, Any]`
- `started_at` / `finished_at`

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
- command line
- output file hashes or at least stable paths + return code

## 25.4 RunResult

**Type:** `eesizer_core.contracts.artifacts.RunResult`

Fields:
- `success: bool`
- `final_metrics: MetricsBundle | None`
- `history: list[dict[str, Any]]`
- `messages: tuple[str, ...]`

`history` is intended to contain step-by-step records (iteration number, patch, metrics, etc.) produced by strategies.
