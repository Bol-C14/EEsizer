from __future__ import annotations

import argparse
from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import NoOptBaselineStrategy


def _build_spec() -> CircuitSpec:
    return CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-opt baseline simulation.")
    parser.add_argument(
        "--netlist",
        type=Path,
        default=Path(__file__).resolve().parent / "rc_lowpass.sp",
        help="Path to a SPICE netlist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Workspace directory for run artifacts.",
    )
    args = parser.parse_args()

    netlist_path = args.netlist
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = _build_spec()

    strategy = NoOptBaselineStrategy()
    if resolve_ngspice_executable(strategy.sim_run_op.ngspice_bin) is None:
        print("ngspice not found; skipping run_noopt_baseline")
        return

    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    ctx = RunContext(workspace_root=args.output)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    print(f"run_dir: {ctx.run_dir()}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"history_len: {len(result.history)}")


if __name__ == "__main__":
    main()
