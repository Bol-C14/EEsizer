from __future__ import annotations

from enum import Enum


class SourceKind(str, Enum):
    spice_netlist = "spice_netlist"
    hdl = "hdl"
    schematic = "schematic"
    graph = "graph"
    embedding = "embedding"
    other = "other"


class SimKind(str, Enum):
    dc = "dc"
    ac = "ac"
    tran = "tran"
    ams = "ams"  # future


class PatchOpType(str, Enum):
    set = "set"
    add = "add"
    mul = "mul"


class StopReason(str, Enum):
    reached_target = "reached_target"
    max_iterations = "max_iterations"
    no_improvement = "no_improvement"
    guard_failed = "guard_failed"
    budget_exhausted = "budget_exhausted"
    policy_stop = "policy_stop"
    error = "error"
