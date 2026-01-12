# 20. Artifact Contracts (General)

Artifacts are the shared language across subprojects.

## 20.1 Required properties

Every artifact MUST satisfy:

1. **Type stability**
   - It has a well-defined schema (dataclass) and a stable meaning.

2. **Serializability**
   - The payload is JSON/YAML serializable *or* the artifact references files on disk.

3. **Stable fingerprint**
   - The artifact can produce a deterministic fingerprint for provenance.
   - Fingerprints MUST be stable across machines and processes.

4. **Immutability by default**
   - Operators produce new artifacts; they do not mutate existing artifacts in-place.

## 20.2 Artifact categories in eesizer_core

| Category | Purpose | Examples |
|---|---|---|
| **Source** | raw user inputs | `CircuitSource` |
| **IR** | parsed/normalized representation | `CircuitIR` |
| **Parameterization** | what can change | `ParamSpace`, `ParamDef` |
| **Action** | proposed changes | `Patch`, `PatchOp` |
| **Simulation** | simulation plan and evidence | `SimPlan`, `SpiceDeck`, `RawSimData` |
| **Metrics** | derived values | `MetricValue`, `MetricsBundle` |
| **Provenance** | audit trail | `Provenance`, `ArtifactFingerprint` |

## 20.3 Artifact naming conventions

- `*_id` fields are stable identifiers (strings).
- `kind` fields MUST be enums when available (see `SimKind`, `SourceKind`).
- Path fields MUST point to files **inside** the run directory.

## 20.4 Forward compatibility

When evolving artifacts:
- Add fields as optional with defaults.
- Avoid changing field meaning.
- If semantics change, bump spec and code version and add an ADR.

