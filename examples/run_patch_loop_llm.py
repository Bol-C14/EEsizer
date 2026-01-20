from __future__ import annotations

import argparse
import os
from pathlib import Path

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.policies import LLMPatchPolicy
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import PatchLoopStrategy


def _build_spec() -> CircuitSpec:
    return CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch-loop with LLMPatchPolicy.")
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
    parser.add_argument("--max-iters", type=int, default=20, help="Max optimization iterations.")
    parser.add_argument("--no-improve", type=int, default=8, help="No-improvement patience.")
    parser.add_argument("--provider", type=str, default="mock", choices=("mock", "openai"), help="LLM backend.")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="LLM max tokens.")
    parser.add_argument("--seed", type=int, default=None, help="LLM seed.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count on JSON parse errors.")
    args = parser.parse_args()

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; export it to use the OpenAI provider.")
        return

    netlist_path = args.netlist
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = _build_spec()

    policy = LLMPatchPolicy(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    strategy = PatchLoopStrategy(policy=policy)
    if resolve_ngspice_executable(strategy.sim_run_op.ngspice_bin) is None:
        print("ngspice not found; skipping run_patch_loop_llm")
        return

    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=args.max_iters, no_improve_patience=args.no_improve),
        notes={"param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]}},
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
