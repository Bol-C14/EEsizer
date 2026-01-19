"""
Stable, explicit API surface for eesizer_core.

Use this module instead of relying on __init__.py re-exports to avoid drift.
"""

from __future__ import annotations

from .contracts import (
    CircuitSource,
    CircuitSpec,
    Objective,
    Constraint,
    ParamDef,
    ParamSpace,
    PatchOp,
    Patch,
    SimRequest,
    SimPlan,
    MetricSpec,
    MetricValue,
    MetricsBundle,
    RunResult,
    TokenLoc,
    Element,
    CircuitIR,
)
from .contracts.enums import SourceKind, SimKind, PatchOpType, StopReason
from .contracts.errors import (
    EEsizerError,
    ContractError,
    ValidationError,
    OperatorError,
    PolicyError,
    SimulationError,
    MetricError,
)
from .contracts.operators import Operator, OperatorResult
from .contracts.policy import Policy, Observation
from .contracts.strategy import Strategy, StrategyConfig, OptimizationBudget
from .contracts.provenance import ArtifactFingerprint, Provenance, RunManifest
from .baselines import LegacyMetricsBaseline
from .strategies import NoOptBaselineStrategy, PatchLoopStrategy, evaluate_objectives
from .policies import FixedSequencePolicy, RandomNudgePolicy, GreedyCoordinatePolicy
from .sim import (
    DeckBuildOperator,
    NgspiceRunOperator,
    CircuitSourceToNetlistBundleOperator,
    SpiceDeck,
    RawSimData,
)
from .metrics import (
    MetricRegistry,
    MetricImplSpec,
    ComputeMetricsOperator,
    DEFAULT_REGISTRY,
    compute_ac_mag_db_at,
    compute_unity_gain_freq,
    compute_dc_vout_last,
    compute_dc_slope,
    compute_tran_rise_time,
)

__all__ = [
    # contracts - artifacts
    "CircuitSource",
    "CircuitSpec",
    "Objective",
    "Constraint",
    "ParamDef",
    "ParamSpace",
    "PatchOp",
    "Patch",
    "SimRequest",
    "SimPlan",
    "MetricSpec",
    "MetricValue",
    "MetricsBundle",
    "RunResult",
    "TokenLoc",
    "Element",
    "CircuitIR",
    # enums
    "SourceKind",
    "SimKind",
    "PatchOpType",
    "StopReason",
    # errors
    "EEsizerError",
    "ContractError",
    "ValidationError",
    "OperatorError",
    "PolicyError",
    "SimulationError",
    "MetricError",
    # protocols
    "Operator",
    "OperatorResult",
    "Policy",
    "Observation",
    "Strategy",
    "StrategyConfig",
    "OptimizationBudget",
    # strategies
    "NoOptBaselineStrategy",
    "PatchLoopStrategy",
    "evaluate_objectives",
    # baselines
    "LegacyMetricsBaseline",
    # policies
    "FixedSequencePolicy",
    "RandomNudgePolicy",
    "GreedyCoordinatePolicy",
    # provenance
    "ArtifactFingerprint",
    "Provenance",
    "RunManifest",
    # sim
    "DeckBuildOperator",
    "NgspiceRunOperator",
    "CircuitSourceToNetlistBundleOperator",
    "SpiceDeck",
    "RawSimData",
    # metrics
    "MetricRegistry",
    "MetricImplSpec",
    "ComputeMetricsOperator",
    "DEFAULT_REGISTRY",
    "compute_ac_mag_db_at",
    "compute_unity_gain_freq",
    "compute_dc_vout_last",
    "compute_dc_slope",
    "compute_tran_rise_time",
]
