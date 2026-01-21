from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...contracts import CircuitSource, MetricsBundle, Patch
from ...contracts.enums import StopReason
from ...contracts.guards import GuardReport


@dataclass
class PatchLoopConfig:
    max_iters: int
    max_sim_runs: int | None
    history_tail_k: int
    max_patch_retries: int
    guard_cfg: dict[str, Any] = field(default_factory=dict)
    validation_opts: dict[str, Any] = field(default_factory=dict)
    apply_opts: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchLoopState:
    current_source: CircuitSource
    current_signature: str
    circuit_ir: Any
    current_metrics: MetricsBundle
    best_source: CircuitSource
    best_metrics: MetricsBundle
    best_score: float
    best_all_pass: bool
    best_iter: int | None
    history: list[dict[str, Any]] = field(default_factory=list)
    no_improve: int = 0
    sim_runs: int = 0
    sim_runs_ok: int = 0
    sim_runs_failed: int = 0


@dataclass
class BaselineResult:
    success: bool
    metrics: MetricsBundle
    guard_report: GuardReport | None
    attempts: list[dict[str, Any]]
    errors: list[str]
    stage_map: dict[str, str]
    warnings: list[str]
    stop_reason: StopReason | None
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int


@dataclass
class AttemptResult:
    attempt: int
    patch: Patch
    guard_report: GuardReport | None
    guard_failures: list[str]
    stage_map: dict[str, str]
    warnings: list[str]
    metrics: MetricsBundle | None
    new_source: CircuitSource
    new_signature: str
    new_circuit_ir: Any
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int
    success: bool


@dataclass
class LoopResult:
    best_source: CircuitSource
    best_metrics: MetricsBundle
    history: list[dict[str, Any]]
    stop_reason: StopReason
    best_score: float
    best_all_pass: bool
    best_iter: int | None
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int
