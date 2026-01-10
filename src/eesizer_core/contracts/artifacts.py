from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .enums import SourceKind, SimKind, PatchOpType, StopReason
from .provenance import ArtifactFingerprint, stable_hash_json, stable_hash_str


Number = Union[int, float]
Scalar = Union[Number, str]  # allow "180n" etc


@dataclass(frozen=True)
class TokenLoc:
    line_idx: int
    token_idx: int
    key: str                 # e.g. "w", "l", "dc"
    raw_token: str           # e.g. "W=1u"
    value_span: Tuple[int, int]  # slice within raw_token after '='


@dataclass(frozen=True)
class Element:
    name: str
    etype: str
    nodes: Tuple[str, ...]
    model_or_subckt: Optional[str] = None
    params: Mapping[str, TokenLoc] = field(default_factory=lambda: MappingProxyType({}))
    line_idx: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class CircuitIR:
    lines: Tuple[str, ...]
    elements: Mapping[str, Element]
    # flattened param map: "M1.w" -> TokenLoc
    param_locs: Mapping[str, TokenLoc] = field(default_factory=lambda: MappingProxyType({}))
    includes: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", MappingProxyType(dict(self.elements)))
        object.__setattr__(self, "param_locs", MappingProxyType(dict(self.param_locs)))


@dataclass(frozen=True)
class CircuitSource:
    kind: SourceKind
    text: str
    name: str = "circuit"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> ArtifactFingerprint:
        # include kind + text only for content identity
        return ArtifactFingerprint(sha256=stable_hash_str(f"{self.kind}:{self.text}"))


@dataclass(frozen=True)
class Objective:
    """One target metric with optional tolerance."""
    metric: str
    target: Optional[float]  # None means "not specified"
    tol: Optional[float] = None  # relative or absolute, strategy decides semantics
    weight: float = 1.0
    sense: str = "ge"  # "ge" >= target, "le" <= target, "eq" approx


@dataclass(frozen=True)
class Constraint:
    """Generic constraint payload (kept abstract in Step1)."""
    kind: str
    data: Dict[str, Any]


@dataclass(frozen=True)
class CircuitSpec:
    """Optimization goal + constraints."""
    objectives: Tuple[Objective, ...] = ()
    constraints: Tuple[Constraint, ...] = ()
    observables: Tuple[str, ...] = ()  # node names, source names, etc
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParamDef:
    """A single tunable parameter definition (whitelist)."""
    param_id: str            # e.g. "M1.w" or "Vbias.dc" or "param:Ibias"
    unit: str = ""           # "m", "V", "A", ...
    lower: Optional[float] = None
    upper: Optional[float] = None
    frozen: bool = False
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ParamSpace:
    params: Tuple[ParamDef, ...]
    _index: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    @staticmethod
    def build(params: Sequence[ParamDef]) -> "ParamSpace":
        idx = {p.param_id: i for i, p in enumerate(params)}
        return ParamSpace(params=tuple(params), _index=MappingProxyType(idx))

    def contains(self, param_id: str) -> bool:
        return param_id in self._index

    def get(self, param_id: str) -> Optional[ParamDef]:
        i = self._index.get(param_id)
        if i is None:
            return None
        return self.params[i]


@dataclass(frozen=True)
class PatchOp:
    param: str
    op: PatchOpType
    value: Scalar
    why: str = ""


@dataclass(frozen=True)
class Patch:
    ops: Tuple[PatchOp, ...] = ()
    stop: bool = False
    notes: str = ""

    def fingerprint(self) -> ArtifactFingerprint:
        # stable identity for patch itself
        payload = {
            "ops": [{"param": o.param, "op": o.op.value, "value": o.value, "why": o.why} for o in self.ops],
            "stop": self.stop,
            "notes": self.notes,
        }
        return ArtifactFingerprint(sha256=stable_hash_json(payload))


@dataclass(frozen=True)
class SimRequest:
    kind: SimKind
    # allow per-sim parameters later (sweep ranges, etc)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimPlan:
    """A plan of simulations to run (minimal in Step1)."""
    sims: Tuple[SimRequest, ...] = ()


@dataclass(frozen=True)
class MetricSpec:
    """Direct lift from legacy/metrics_contract.py but generalized."""
    name: str
    unit: str
    sim: SimKind
    required_files: Tuple[str, ...] = ()
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: Optional[float]
    unit: str = ""
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsBundle:
    values: Dict[str, MetricValue] = field(default_factory=dict)

    def get(self, name: str) -> Optional[MetricValue]:
        return self.values.get(name)


@dataclass
class RunResult:
    """Final output of a strategy run."""
    best_source: Optional[CircuitSource] = None
    best_metrics: MetricsBundle = field(default_factory=MetricsBundle)
    history: List[Dict[str, Any]] = field(default_factory=list)  # per-iteration records
    stop_reason: Optional[StopReason] = None
    notes: Dict[str, Any] = field(default_factory=dict)
