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
from eesizer_core.strategies import GridSearchStrategy


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


def _load_benchmark(bench_name: str) -> tuple[CircuitSource, CircuitSpec, Path, list[str]]:
    bench_dir = BENCH_ROOT / bench_name
    if not bench_dir.exists():
        raise FileNotFoundError(f"unknown benchmark '{bench_name}'")

    bench_payload = _read_json(bench_dir / "bench.json")
    top_netlist = bench_payload.get("top_netlist", "bench.sp")
    spec_name = bench_payload.get("spec", "spec.json")

    netlist_path = bench_dir / top_netlist
    spec_path = bench_dir / spec_name

    recommended_knobs = []
    for item in bench_payload.get("recommended_knobs", []) or []:
        if isinstance(item, str) and item.strip():
            recommended_knobs.append(item.strip())

    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        name=str(bench_payload.get("name", bench_name)),
        metadata={"base_dir": bench_dir.parent},
    )
    spec = _build_spec(_read_json(spec_path))
    return source, spec, bench_dir, recommended_knobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid search on a benchmark.")
    parser.add_argument("--bench", required=True, help="Benchmark name (rc, ota, opamp3).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Workspace directory for run artifacts.",
    )
    parser.add_argument("--max-iters", type=int, default=21, help="Total iterations (baseline + candidates).")
    parser.add_argument("--levels", type=int, default=10, help="Levels per parameter.")
    parser.add_argument("--span-mul", type=float, default=10.0, help="Span multiplier when bounds are missing.")
    parser.add_argument("--scale", type=str, default="log", choices=("log", "linear"), help="Level spacing.")
    parser.add_argument("--mode", type=str, default="coordinate", choices=("coordinate", "factorial"), help="Grid mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K to record.")
    parser.add_argument("--max-params", type=int, default=8, help="Max params to sweep.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates (optional).")
    parser.add_argument("--include-nominal", action="store_true", help="Include nominal in candidate list.")
    args = parser.parse_args()

    strategy = GridSearchStrategy()
    if resolve_ngspice_executable(strategy.sim_run_op.ngspice_bin) is None:
        print("ngspice not found; skipping run_grid_search_bench")
        return

    source, spec, bench_dir, recommended_knobs = _load_benchmark(args.bench)

    grid_notes: dict[str, Any] = {
        "mode": args.mode,
        "levels": args.levels,
        "span_mul": args.span_mul,
        "scale": args.scale,
        "top_k": args.top_k,
        "max_params": args.max_params,
        "max_candidates": args.max_candidates,
        "include_nominal": args.include_nominal,
        "param_select_policy": "recommended",
    }
    if recommended_knobs:
        grid_notes["recommended_knobs"] = recommended_knobs

    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=args.max_iters, no_improve_patience=1),
        notes={"grid_search": grid_notes},
    )
    ctx = RunContext(workspace_root=args.out)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    print(f"bench_dir: {bench_dir}")
    print(f"recommended_knobs: {len(recommended_knobs)}")
    print(f"run_dir: {ctx.run_dir()}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"best_score: {result.notes.get('best_score')}")


if __name__ == "__main__":
    main()
