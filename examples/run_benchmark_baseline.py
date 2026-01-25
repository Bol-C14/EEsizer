from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from eesizer_core.contracts import CircuitSource, CircuitSpec, Constraint, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.metrics.aliases import canonicalize_metric_name
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import NoOptBaselineStrategy


BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


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

    observables = []
    for obs in payload.get("observables", []) or []:
        if isinstance(obs, str) and obs.strip():
            observables.append(obs.strip())

    notes = payload.get("notes", {})
    if not isinstance(notes, Mapping):
        notes = {}

    return CircuitSpec(
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        observables=tuple(observables),
        notes=dict(notes),
    )


def _load_benchmark(bench_name: str) -> tuple[CircuitSource, CircuitSpec, Path]:
    bench_dir = BENCH_ROOT / bench_name
    if not bench_dir.exists():
        raise FileNotFoundError(f"unknown benchmark '{bench_name}'")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline for a benchmark.")
    parser.add_argument("--bench", required=True, help="Benchmark name (rc, ota, opamp3).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Workspace directory for run artifacts.",
    )
    args = parser.parse_args()

    strategy = NoOptBaselineStrategy()
    if resolve_ngspice_executable(strategy.sim_run_op.ngspice_bin) is None:
        print("ngspice not found; skipping run_benchmark_baseline")
        return

    source, spec, bench_dir = _load_benchmark(args.bench)

    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    ctx = RunContext(workspace_root=args.out)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    print(f"bench_dir: {bench_dir}")
    print(f"run_dir: {ctx.run_dir()}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"history_len: {len(result.history)}")


if __name__ == "__main__":
    main()
