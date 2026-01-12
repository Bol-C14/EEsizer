# 40. Patch-only Protocol (Safety Contract)

This protocol exists to stop a common failure mode in LLM-driven circuit optimisation:

- the model edits the netlist freely,
- which silently changes circuit function,
- which destroys reproducibility and breaks downstream checks.

**Rule:** policies output `Patch` only. They MUST NOT output a full modified netlist.

## 40.1 The contract

A policy takes:
- the original circuit source
- the parameter space (what may change)
- observations (metrics, logs)

And returns:
- a Patch: `{ ops: [{param_id, value, mode}], metadata... }`

## 40.2 Enforcement points

The protocol is enforced at three levels:
1. JSON schema validation (structure)
2. param space validation (only allowed params, bounds, frozen)
3. topology validation (no structure drift)

## 40.3 Recommended LLM IO pattern

Provide the model:
- the circuit (read-only)
- the param space (whitelist)
- current metrics and targets
- constraints and budgets

Require the model to respond with:
- **JSON only** matching the patch schema

## 40.4 Why this helps explainability

A patch is a small diff:
- you can attribute metric changes to a handful of parameter updates
- you can graph search traces and convergence
- you can compare strategies independent of the policy backend

## 40.5 Exceptions

Exceptions MUST be explicit and reviewed (ADR):
- topology-changing edits for circuit synthesis or architecture search
- those workflows require different invariants and stronger checks
