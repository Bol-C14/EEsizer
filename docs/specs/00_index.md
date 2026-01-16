# 00. Spec Index

This index is the navigation hub for the **EEsizer Core** contract.

## Core model

- [10_core_model.md](10_core_model.md): Universe, Artifact, Operator, Policy, Strategy, Run.
- [11_runtime_layout.md](11_runtime_layout.md): where files go (run directories, stage directories).

## Data contracts (Artifacts)

- [20_artifact_contracts.md](20_artifact_contracts.md): artifact taxonomy and invariants.
- [21_artifact_circuit.md](21_artifact_circuit.md): CircuitSource, CircuitIR, ParamSpace.
- [22_artifact_patch.md](22_artifact_patch.md): Patch and PatchOp.
- [23_artifact_simulation.md](23_artifact_simulation.md): SimPlan, SpiceDeck, RawSimData.
- [24_artifact_metrics.md](24_artifact_metrics.md): MetricSpec, MetricsBundle.
- [25_artifact_provenance.md](25_artifact_provenance.md): Provenance, fingerprints, stable hashing.
- [26_artifact_guards.md](26_artifact_guards.md): GuardCheck, GuardReport.

## Execution contracts (Operators)

- [30_operator_contracts.md](30_operator_contracts.md): operator interface, errors, determinism.
- [31_operator_deck_build.md](31_operator_deck_build.md)
- [32_operator_ngspice_run.md](32_operator_ngspice_run.md)
- [33_operator_metrics_compute.md](33_operator_metrics_compute.md)
- [34_operator_patch_apply.md](34_operator_patch_apply.md)
- [35_operator_guards.md](35_operator_guards.md)

## Decision contracts (Policy + Strategy)

- [40_patch_only_protocol.md](40_patch_only_protocol.md): the safety rule that bans full-netlist rewriting.
- [41_policy_strategy_contracts.md](41_policy_strategy_contracts.md): what decisions look like, how loops stop.

## Engineering standards

- [60_coding_standards.md](60_coding_standards.md): code style, docs rules, tests.
- [61_extension_guide.md](61_extension_guide.md): add a simulator, add a metric, add a policy.

## Step 4: gaps, fixes, acceptance

- [90_step4_fixlist_and_acceptance.md](90_step4_fixlist_and_acceptance.md)
- [91_step4_definition_of_done.md](91_step4_definition_of_done.md)
