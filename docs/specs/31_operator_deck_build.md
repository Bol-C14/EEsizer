# 31. Operator: DeckBuildOperator

**Code:** `eesizer_core.sim.deck_builder.DeckBuildOperator`

## Purpose

Build a `SpiceDeck` by injecting a `.control/.endc` block for one analysis kind.

This operator is the *only* place that should generate ngspice control scripts.

## Inputs

The operator accepts one of the following netlist sources:

1) `circuit_source: CircuitSource` (recommended)
- must have `kind == SourceKind.spice_netlist`
- `metadata` may include `base_dir` for `.include` resolution

2) `netlist_bundle: NetlistBundle`

3) `circuit_ir: CircuitIR`

4) `netlist_text: str` (discouraged, low context)

Additional required inputs:
- `sim_plan: SimPlan`  
- optional `sim_kind: SimKind` (select which sim from the plan)

Optional inputs:
- `base_dir: str | Path` (used when not inferable from source)

## Outputs

- `deck: SpiceDeck`
- `output_nodes: tuple[str, ...]`

## Wrdata outputs (standardized)

- **AC**: writes `ac.csv` with columns `frequency`, `real(v(<node>))`, `imag(v(<node>))` for each requested output node. For multiple nodes, the scale column repeats with suffixes (e.g., `frequency`, `frequency_1`, ...).
- **DC**: writes `dc.csv` with columns `v(<sweep_node>)`, `v(<node>)` for each output node; scale repeats per node as `v(<sweep_node>)`, `v(<sweep_node>)_1`, ...
- **TRAN**: writes `tran.csv` with columns `time`, `v(<node>)` for each output node; scale repeats per node as `time`, `time_1`, ...

## Validation rules

- input netlist MUST NOT already contain `.control/.endc` blocks.
- `sim_plan` MUST include a request for the selected simulation kind.
- output nodes MUST be non-empty strings.

## Provenance (recommended)

DeckBuildOperator SHOULD record:
- hash of the input netlist text
- hash of the sim plan
- selected sim kind
- deck hash (kind, expected outputs)
