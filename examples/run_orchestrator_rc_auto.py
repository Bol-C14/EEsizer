from __future__ import annotations

import argparse
from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import MultiAgentOrchestratorStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent orchestrator on RC lowpass.")
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
    parser.add_argument("--robust", action="store_true", help="Use corner_search for robustness.")
    args = parser.parse_args()

    netlist_path = args.netlist
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )

    # Provide an empty spec: the orchestrator will synthesize a default (RC heuristic).
    spec = CircuitSpec()

    strategy = MultiAgentOrchestratorStrategy()
    # Orchestrator may execute an ngspice-backed search strategy.
    # If ngspice is missing, print a friendly message.
    if resolve_ngspice_executable("ngspice") is None:
        print("ngspice not found; skipping run_orchestrator_rc_auto")
        return

    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=args.max_iters, no_improve_patience=1),
        notes={"orchestrator": {"robust": args.robust}},
    )
    ctx = RunContext(workspace_root=args.output)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    best_mv = result.best_metrics.get("ac_mag_db_at_1k")

    print(f"orchestrator_run_dir: {ctx.run_dir()}")
    print(f"child_run_dir: {result.notes.get('child_run', {}).get('run_dir')}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"best_score: {result.notes.get('child_run', {}).get('best_score')}")
    if best_mv is not None:
        print(f"best_ac_mag_db_at_1k: {best_mv.value} {best_mv.unit}")


if __name__ == "__main__":
    main()
