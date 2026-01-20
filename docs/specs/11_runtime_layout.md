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

### Required run artifacts (Step 7)

Each run SHOULD also materialize the following subfolders/files:

```
runs/<run_id>/
  run_manifest.json
  inputs/
    source.sp
    spec.json
    param_space.json
    cfg.json
    signature.txt
  llm/
    llm_i001_a00/
      request.json
      prompt.txt
      response.txt
      parsed_patch.json
      parse_error.txt
      call_error.txt
  history/
    iterations.jsonl
    summary.json
  provenance/
    operator_calls.jsonl
  best/
    best.sp
    best_metrics.json
```

All paths referenced in manifest/history SHOULD be relative to the run directory.

### JSONL schemas (minimal)

`history/iterations.jsonl` lines SHOULD include:
- `iteration`, `patch`, `metrics`, `score`, `sim_stages`, `guard`, `attempts`

`provenance/operator_calls.jsonl` lines SHOULD include:
- `operator`, `version`, `start_time`, `end_time`, `inputs`, `outputs`, `command`, `notes`

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
- When a strategy retries within the same iteration, stage names SHOULD include an attempt suffix (e.g., `ac_i003_a01`) to avoid overwriting artifacts.
- LLM stages SHOULD use `llm/llm_i{iter}_a{attempt}` and append retry suffixes (`_r01`) when parsing is retried.

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

## 11.4 Run manifest

A `run_manifest.json` is REQUIRED for end-to-end strategies and should contain:
- run metadata (time, environment, policy/strategy)
- inputs (source/spec/param_space hashes + signature)
- file index for inputs/history/provenance/best
- if LLM is used, a list of `llm/` artifact paths
- result summary (stop reason, best iter/score, sim runs total/ok/failed)

Template:
- `docs/specs/schemas/run_manifest.example.json` (mirrored from `docs/templates/run_manifest.example.json`)

## 11.5 Content addressing

When an artifact payload is large (waveforms), the artifact SHOULD reference paths under the run dir and fingerprint them.

Example: `RawSimData.outputs: dict[str, Path]` must point to files inside the stage dir.
