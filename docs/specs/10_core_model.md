# 10. Core Model (Universe -> Run)

This doc defines the top-level abstraction used across EEsizer, RTL-Pilot, and Tasks 1–4.

## 10.1 Universe of representations

Let **U** be the set of all representations of a design or its evidence.

Examples (non-exhaustive):
- SPICE netlist text
- HDL/RTL source
- schematic graph / net graph
- symbolic IR
- a vector embedding
- an image of a schematic
- waveform files (CSV/RAW)
- coverage databases and formal proof artifacts

**Key idea:** we do not hard-code toolchains into the architecture. Tools are functions that transform elements of **U**.

## 10.2 Artifact

An **Artifact** is a typed element of **U** with a stable identity.

Formally, an artifact `a` is a tuple:

- `type(a)` (schema / dataclass)
- `value(a)` (payload)
- `fp(a)` (fingerprint)

where `fp(a)` is a stable hash over the payload (or over content-addressed files).

### Artifact invariants

Artifacts MUST:
- be serializable (or reference serialized files)
- be hashable (stable fingerprint)
- be immutable-by-default (new artifact per transformation)

## 10.3 Operator

An **Operator** is a reusable transformation with provenance:

```
O: (A^n, C) -> (A^m, P)
```

- `A` is the space of artifacts
- `C` is an execution context (run directory, environment)
- `P` is provenance (hashes, versions, commands, timestamps)

Operators MAY call external tools (ngspice, simulators, formal engines) but they MUST:
- write only under `ctx.run_dir()`
- declare and record their side effects via provenance
- fail loudly with structured errors

## 10.4 Policy

A **Policy** is a decision function that proposes structured actions.

```
π: Obs -> Action
```

- `Obs` is an observation made of artifacts (metrics, logs, constraints)
- `Action` is usually a `Patch` (parameter deltas) or a plan proposal

Policies MUST NOT directly execute external tools or write files outside the run logs.

Policies can be:
- LLM-based
- RL / neural
- Bayesian optimisation
- heuristics or hand-coded rules

## 10.5 Strategy

A **Strategy** composes operators + policies into a workflow:

```
S: (source, spec, cfg, ctx) -> RunResult
```

It is responsible for:
- stop conditions (budget, convergence, target met)
- guards (reject unsafe actions)
- composing operators (build deck -> run -> metrics -> decide -> patch -> repeat)

## 10.6 Why this split exists

Traditional EDA flows are fixed pipelines:

- great for repeatability,
- painful to extend,
- glue-heavy when each team uses different tools.

This model aims for:
- a shared coordinate system (Artifacts)
- reusable tool wrappers (Operators)
- replaceable brains (Policies)
- explicit workflow logic (Strategies)

The result is less glue, more reuse, and better auditability.
