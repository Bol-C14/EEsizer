from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, Constraint, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.metrics.aliases import canonicalize_metric_name
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import NoOptBaselineStrategy


BENCH_ROOT = Path(__file__).resolve().parent.parent / "benchmarks"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_spec(payload: Mapping[str, Any]) -> CircuitSpec:
    objectives = []
    for obj in payload.get("objectives", []) or []:
        if not isinstance(obj, Mapping):
            continue
        metric = obj.get("metric")
        if not metric:
            continue
        objectives.append(
            Objective(
                metric=canonicalize_metric_name(str(metric)),
                target=obj.get("target"),
                tol=obj.get("tol"),
                weight=obj.get("weight", 1.0),
                sense=obj.get("sense", "ge"),
            )
        )

    constraints = []
    for item in payload.get("constraints", []) or []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        data = item.get("data", {})
        if not kind or not isinstance(data, Mapping):
            continue
        constraints.append(Constraint(kind=str(kind), data=dict(data)))

    notes = payload.get("notes", {})
    if not isinstance(notes, Mapping):
        notes = {}

    return CircuitSpec(objectives=tuple(objectives), constraints=tuple(constraints), notes=dict(notes))


def _load_benchmark(bench_name: str) -> tuple[CircuitSource, CircuitSpec, Path]:
    bench_dir = BENCH_ROOT / bench_name
    bench_payload = _read_json(bench_dir / "bench.json")
    top_netlist = bench_payload.get("top_netlist", "bench.sp")
    spec_name = bench_payload.get("spec", "spec.json")

    netlist_path = bench_dir / top_netlist
    spec_path = bench_dir / spec_name

    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        name=str(bench_payload.get("name", bench_name)),
        metadata={"base_dir": bench_dir.parent},
    )
    spec = _build_spec(_read_json(spec_path))
    return source, spec, bench_dir


@pytest.mark.integration
def test_step2_ota_metrics_present(tmp_path: Path) -> None:
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    source, spec, _ = _load_benchmark("ota")
    ctx = RunContext(workspace_root=tmp_path / "ota")
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    strategy = NoOptBaselineStrategy()

    strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

    metrics_path = ctx.run_dir() / "best" / "best_metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    ugbw = metrics_payload.get("ugbw_hz", {}).get("value")
    pm = metrics_payload.get("phase_margin_deg", {}).get("value")
    power = metrics_payload.get("power_w", {}).get("value")

    assert ugbw is not None and ugbw > 0
    assert pm is not None and 0 < pm < 180
    assert power is not None and power > 0
