# 23. Simulation Artifacts

Simulation artifacts define:
- what to simulate (`SimPlan`)
- what to run (`SpiceDeck`)
- what evidence comes back (`RawSimData`)

## 23.1 SimKind

**Type:** `eesizer_core.contracts.enums.SimKind`

Values in current code:
- `ac`, `dc`, `tran`, `ams`

For EEsizer-core Step 4, `ac/dc/tran` are implemented; `ams` is reserved for future mixed-signal.

## 23.2 SimRequest and SimPlan

**SimRequest type:** `eesizer_core.contracts.artifacts.SimRequest`
- `kind: SimKind`
- `params: dict[str, Any]`  
  Simulator parameters, such as sweep and output nodes.

**SimPlan type:** `eesizer_core.contracts.artifacts.SimPlan`
- `sims: tuple[SimRequest, ...]`

**Invariants**
- A strategy MUST specify the SimPlan for a run.
- A SimPlan MAY include multiple simulations (e.g., dc + ac + tran). Operators choose one at a time.

## 23.3 NetlistBundle

**Type:** `eesizer_core.sim.artifacts.NetlistBundle`

This is a convenience wrapper when a netlist is paired with a base directory used for `.include` resolution.

Fields:
- `text: str`
- `base_dir: Path`
- `include_files: tuple[Path, ...]` (optional explicit include list)
- `extra_search_paths: tuple[Path, ...]`

## 23.4 SpiceDeck

**Type:** `eesizer_core.sim.artifacts.SpiceDeck`

Fields:
- `text: str`  
  Netlist with an injected `.control ... .endc` block.
- `kind: SimKind`
- `expected_outputs: dict[str, str]`  
  Mapping from logical key to **relative path** (within a stage dir). Example: `{ "ac_csv": "ac.csv" }`.
- `expected_outputs_meta: dict[str, tuple[str, ...]]`  
  Per-output ordered column names for wrdata outputs.
- `workdir: Path | None`  
  Directory used as ngspice CWD when running (defaults to stage dir).

**Invariants**
- `text` MUST NOT contain `.control/.endc` before injection. Input sanitization MUST reject it.
- Every `expected_outputs` value MUST be a relative path that does not escape the stage directory.

## 23.5 RawSimData

**Type:** `eesizer_core.sim.artifacts.RawSimData`

Fields:
- `kind: SimKind`
- `run_dir: Path` (the stage directory)
- `outputs: dict[str, Path]` (absolute paths)
- `outputs_meta: dict[str, tuple[str, ...]]`
- `log_path: Path`
- `cmdline: list[str]`
- `returncode: int`
- `stdout_tail: str`
- `stderr_tail: str`

**Invariants**
- Every `outputs[*]` path MUST exist.
- Every `outputs[*]` path MUST be within `run_dir`.
