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
- `timestamp_start: str | None`
- `timestamp_end: str | None`
- `inputs: dict[str, Any]` (hashes + signature)
- `environment: dict[str, Any]` (python/platform/tool/policy metadata)
- `files: dict[str, str]` (relative paths under run_dir)
- `result_summary: dict[str, Any]` (best iter/score, stop reason, sim runs)
- `result_summary: dict[str, Any]` (best iter/score, stop reason, sim runs total/ok/failed)
- `tool_versions: dict[str, str]`
- `env: dict[str, str]`
- `notes: dict[str, Any]`

Helpers:
- `to_dict() -> dict` returns a JSON-ready payload.
- `save_json(path: Path)` writes the manifest (parents created).

## 25.5 RunResult

**Type:** `eesizer_core.contracts.artifacts.RunResult`

Fields:
- `best_source: CircuitSource | None`
- `best_metrics: MetricsBundle`
- `history: list[dict[str, Any]]` (per-iteration records)
- `stop_reason: StopReason | None` (includes `baseline_noopt` and `baseline_legacy` for baseline runs)
- `notes: dict[str, Any]`

## 25.6 operator_calls.jsonl

Each line is a JSON object derived from `Provenance`, containing at minimum:
- `operator`, `version`
- `start_time`, `end_time`, `duration_s`
- `inputs`, `outputs` (sha256 maps)
- `command` (if applicable)
- `notes` (tool metadata, cwd, return code)

### LLM call expectations

For `llm_call`, `notes` SHOULD include:
- `provider`, `model`, `temperature`, `max_tokens` (if set)
- `response_schema_name` (if provided)
- `usage` metadata when available

Inputs SHOULD include a prompt or request hash and outputs SHOULD include a response hash.
