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
BENCHES = ("rc", "ota", "opamp3")


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


def _header_line(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


@pytest.mark.integration
def test_benchmarks_baseline_ngspice(tmp_path: Path) -> None:
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    strategy = NoOptBaselineStrategy()

    for bench in BENCHES:
        source, spec, _ = _load_benchmark(bench)
        ctx = RunContext(workspace_root=tmp_path / bench)
        cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
        strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)

        run_dir = ctx.run_dir()
        ac_path = run_dir / "ac_i000_a00" / "ac.csv"
        dc_path = run_dir / "dc_i000_a00" / "dc.csv"
        assert ac_path.exists()
        assert dc_path.exists()

        ac_header = _header_line(ac_path).lower()
        dc_header = _header_line(dc_path).lower()
        assert "v(vout)" in ac_header
        assert "v(vout)" in dc_header
        assert "i(vdd)" in dc_header
