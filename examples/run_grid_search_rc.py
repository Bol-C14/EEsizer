from __future__ import annotations

import argparse
from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import GridSearchStrategy


def _build_spec() -> CircuitSpec:
    return CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid search on RC lowpass.")
    parser.add_argument(
        "--netlist",
        type=Path,
        default=Path(__file__).resolve().parent / "circuits" / "rc_lowpass.sp",
        help="Path to a SPICE netlist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Workspace directory for run artifacts.",
    )
    parser.add_argument("--max-iters", type=int, default=21, help="Total iterations (baseline + candidates).")
    parser.add_argument("--levels", type=int, default=10, help="Levels per parameter.")
    parser.add_argument("--span-mul", type=float, default=10.0, help="Span multiplier when bounds are missing.")
    parser.add_argument("--scale", type=str, default="log", choices=("log", "linear"), help="Level spacing.")
    parser.add_argument(
        "--mode",
        type=str,
        default="coordinate",
        choices=("coordinate", "factorial"),
        help="Grid mode.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K to record.")
    parser.add_argument("--stop-on-first-pass", action="store_true", help="Stop when a candidate passes.")
    args = parser.parse_args()

    netlist_path = args.netlist
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = _build_spec()

    strategy = GridSearchStrategy()
    if resolve_ngspice_executable(strategy.sim_run_op.ngspice_bin) is None:
        print("ngspice not found; skipping run_grid_search_rc")
        return

    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=args.max_iters, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "grid_search": {
                "mode": args.mode,
                "levels": args.levels,
                "span_mul": args.span_mul,
                "scale": args.scale,
                "top_k": args.top_k,
                "stop_on_first_pass": args.stop_on_first_pass,
            },
        },
    )
    ctx = RunContext(workspace_root=args.output)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    best_mv = result.best_metrics.get("ac_mag_db_at_1k")
    print(f"run_dir: {ctx.run_dir()}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"best_score: {result.notes.get('best_score')}")
    if best_mv is not None:
        print(f"best_ac_mag_db_at_1k: {best_mv.value} {best_mv.unit}")


if __name__ == "__main__":
    main()
