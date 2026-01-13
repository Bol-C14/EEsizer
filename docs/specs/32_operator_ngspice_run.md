# 32. Operator: NgspiceRunOperator

**Code:** `eesizer_core.sim.ngspice_runner.NgspiceRunOperator`

## Purpose

Run ngspice in batch mode for a prepared `SpiceDeck`.

## Inputs

- `deck: SpiceDeck` (required)
- `stage: str` (optional)

Context (`ctx`): MUST be a `RunContext` with `run_dir()`.

If `stage` is omitted, the operator defaults to `deck.kind.value`.

## Outputs

- `raw_data: RawSimData`
- `deck_path: Path`
- `log_path: Path`

## Stage layout

The operator creates:

```
<run_dir>/<stage>/
  deck_<kind>.sp
  ngspice_<kind>.log
  <wrdata outputs>
```

All expected outputs MUST be resolved to files under the stage directory.

## Failure modes

- raises `SimulationError` if ngspice is missing, times out, returns non-zero (configurable), or expected outputs are missing.
- raises `ValidationError` for unsafe paths or invalid stage names.

## Provenance

The operator MUST record:
- deck kind hash and deck text hash
- expected outputs mapping hash
- command line used
- return code
- ngspice resolved path (`shutil.which`) in `notes["ngspice_path"]`
- best-effort ngspice version (`ngspice -v`) in `notes["ngspice_version"]` (non-fatal if probing fails)
- working directory used for ngspice (`notes["cwd"]`)
- stdout/stderr tails (`notes["stdout_tail"]` / `notes["stderr_tail"]`)
- log path hash in outputs (`log_path`)
