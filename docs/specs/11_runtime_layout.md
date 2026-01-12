# 11. Runtime Layout and File Discipline

This doc specifies **where files go** and the rules that prevent runaway side effects.

## 11.1 Run directory

A *run* is identified by a `run_id` and owns a root folder:

```
<out_root>/runs/<run_id>/
```

In code, `RunContext.run_dir()` MUST return this root path.

### Run dir invariants

- Operators MUST NOT write outside the run directory.
- Every operator that writes files MUST write into a **stage directory** under the run directory.

## 11.2 Stage directory

A stage is a named subfolder under the run dir:

```
runs/<run_id>/<stage_name>/
```

Examples:
- `ac/`
- `dc/`
- `tran/`
- `corner_vdd_low_temp_high/`

### Stage name rules

- Stage names MUST be filesystem-safe.
- Stage names SHOULD be unique within a run.

## 11.3 Recommended stage contents

A stage SHOULD contain:
- input snapshots (deck, patched netlist)
- tool logs
- tool outputs (CSV/RAW)
- intermediate parse artifacts when helpful

For ngspice stages, the following names are recommended:
- `deck_<kind>.sp`
- `ngspice_<kind>.log`
- output files produced by `wrdata` (for example `ac.csv`, `tran.csv`)

## 11.4 Run manifest (planned target)

The repository already includes a template:
- `docs/specs/schemas/run_manifest.example.json (mirrored from docs/templates/run_manifest.example.json)`

A `run_manifest.json` is RECOMMENDED for end-to-end strategies. It should contain:
- run metadata (time, git commit, environment)
- inputs (source/spec hashes)
- all operator provenance entries
- final metrics and decision trace

## 11.5 Content addressing

When an artifact payload is large (waveforms), the artifact SHOULD reference paths under the run dir and fingerprint them.

Example: `RawSimData.outputs: dict[str, Path]` must point to files inside the stage dir.

