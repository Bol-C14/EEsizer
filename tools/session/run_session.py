from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from eesizer_core.analysis.session_report import build_meta_report
from eesizer_core.contracts import CircuitSource, CircuitSpec, Constraint, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.metrics.aliases import canonicalize_metric_name
from eesizer_core.operators.session_llm_advice import SessionLLMAdviseOperator
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.session_store import SessionStore
from eesizer_core.strategies.interactive_session import InteractiveSessionStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmarks"


def _read_json(path: Path) -> Any:
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


def _load_benchmark(bench_name: str) -> tuple[CircuitSource, CircuitSpec, list[str]]:
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
    return source, spec, recommended_knobs


def _load_delta(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        payload = _read_json(path)
        return payload if isinstance(payload, dict) else None
    payload = json.loads(value)
    return payload if isinstance(payload, dict) else None


def _mock_measure_fn(_src: CircuitSource, iter_idx: int) -> MetricsBundle:
    mb = MetricsBundle()
    mb.values["ugbw_hz"] = MetricValue(name="ugbw_hz", value=1e6 + 1e5 * iter_idx, unit="Hz")
    mb.values["phase_margin_deg"] = MetricValue(name="phase_margin_deg", value=70.0, unit="deg")
    mb.values["power_w"] = MetricValue(name="power_w", value=1e-3 + 1e-5 * iter_idx, unit="W")
    return mb


def cmd_new(args: argparse.Namespace) -> None:
    source, spec, recommended_knobs = _load_benchmark(args.bench)

    notes: dict[str, Any] = {
        "session": {"run_baseline": not args.skip_baseline and not args.mock},
        "grid_search": {
            "mode": args.mode,
            "levels": args.levels,
            "span_mul": args.span_mul,
            "scale": args.scale,
            "continue_after_baseline_pass": args.continue_after_baseline_pass,
            "param_select_policy": "recommended",
            "recommended_knobs": recommended_knobs,
        },
        "corner_validate": {
            "candidates_source": args.candidates_source,
            "corners": args.corners,
            "override_mode": args.override_mode,
            "clamp_corner_overrides": not args.no_clamp,
            "top_m": args.top_m,
        },
    }

    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=args.max_iters, no_improve_patience=1),
        seed=args.seed,
        notes=notes,
    )

    strat = InteractiveSessionStrategy(measure_fn=_mock_measure_fn if args.mock else None)
    session_ctx = strat.start_session(
        bench_id=args.bench,
        source=source,
        spec=spec,
        cfg=cfg,
        workspace_root=args.out,
        run_to_phase=args.run_to_phase,
        reason=args.reason or "new",
    )
    print(f"session_run_dir: {session_ctx.run_dir()}")
    print("meta_report: " + str(session_ctx.run_dir() / "session" / "meta_report.md"))


def cmd_continue(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    state = store.load_session_state()
    # Source is always loaded from bench at continue time (avoids storing large netlist copies in session).
    source, _, _ = _load_benchmark(state.bench_id)

    spec_delta = _load_delta(args.spec_delta)
    cfg_delta = _load_delta(args.cfg_delta)

    strat = InteractiveSessionStrategy(measure_fn=_mock_measure_fn if args.mock else None)
    strat.continue_session(
        session_run_dir=args.session_run_dir,
        source=source,
        spec_delta=spec_delta,
        cfg_delta=cfg_delta,
        next_phase=args.next_phase,
        reason=args.reason or "continue",
    )
    print("meta_report: " + str(Path(args.session_run_dir) / "session" / "meta_report.md"))


def cmd_inspect(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    report = build_meta_report(store)
    store.recorder.write_text("session/meta_report.md", report)
    print("meta_report: " + str(Path(args.session_run_dir) / "session" / "meta_report.md"))


def _session_ctx_for_run_dir(session_run_dir: Path) -> RunContext:
    store = SessionStore(session_run_dir)
    state = store.load_session_state()
    runs_dir = session_run_dir.parent
    workspace_root = runs_dir.parent
    return RunContext(workspace_root=workspace_root, run_id=state.session_id, seed=state.seed)


def cmd_advise(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    state = store.load_session_state()

    llm_cfg: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    inputs: dict[str, Any] = {
        "session_run_dir": args.session_run_dir,
        "llm_config": llm_cfg,
        "max_repairs": args.max_repairs,
    }
    if args.mock_llm:
        inputs["llm_config"] = {**llm_cfg, "provider": "mock"}
        inputs["mock_response"] = json.dumps(
            {
                "summary": "Mock insights summary.",
                "tradeoffs": [{"x": "power_w", "y": "ugbw_hz", "pattern": "mock", "evidence": ["search/topk.json"]}],
                "sensitivity_rank": [{"param_id": "m1.w", "impact": "high", "evidence": ["insights/sensitivity.json"]}],
                "robustness_notes": [{"worst_corner_id": "all_low", "why": "mock", "evidence": ["search/robust_topk.json"]}],
                "recommended_actions": [{"action_type": "increase_levels", "rationale": "mock"}],
            }
        )
        inputs["mock_plan"] = json.dumps(
            {
                "options": [
                    {
                        "title": "Mock Plan A",
                        "intent": "mock",
                        "spec_delta": None,
                        "cfg_delta": {"notes": {"grid_search": {"levels": 10}}},
                        "plan": None,
                        "expected_effects": ["mock"],
                        "risks": ["mock"],
                        "budget_estimate": {"iters": 10, "corners": 0},
                    }
                ]
            }
        )

    op = SessionLLMAdviseOperator()
    ctx = _session_ctx_for_run_dir(args.session_run_dir)
    result = op.run(inputs, ctx=ctx)
    outputs = result.outputs
    print(f"session_id: {state.session_id}")
    print(f"advice_rev: {outputs.get('advice_rev')}")
    print(f"advice_dir: {outputs.get('advice_dir')}")
    print("meta_report: " + str(Path(args.session_run_dir) / "session" / "meta_report.md"))


def cmd_show_advice(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    state = store.load_session_state()
    rev = args.advice_rev if args.advice_rev is not None else state.latest_advice_rev
    if rev is None:
        raise SystemExit("no advice_rev available; run advise first")
    advice_dir = store.advice_dir(rev)
    proposal_path = advice_dir / "proposal.json"
    proposal = _read_json(proposal_path)
    if not isinstance(proposal, dict):
        raise SystemExit(f"invalid proposal.json at {proposal_path}")
    options = proposal.get("options") or []
    if not isinstance(options, list):
        raise SystemExit("proposal.options must be a list")
    print(f"advice_rev: {rev}")
    for idx, opt in enumerate(options):
        if not isinstance(opt, Mapping):
            continue
        print(f"[{idx}] {opt.get('title')} — {opt.get('intent')}")


def cmd_reject(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    state = store.load_session_state()
    rev = args.advice_rev if args.advice_rev is not None else state.latest_advice_rev
    if rev is None:
        raise SystemExit("no advice_rev available; run advise first")
    store.mark_advice_decision(rev=rev, decision="rejected", reason=args.reason)
    print("rejected.")
    print("meta_report: " + str(Path(args.session_run_dir) / "session" / "meta_report.md"))


def cmd_accept(args: argparse.Namespace) -> None:
    store = SessionStore(args.session_run_dir)
    state = store.load_session_state()
    rev = args.advice_rev if args.advice_rev is not None else state.latest_advice_rev
    if rev is None:
        raise SystemExit("no advice_rev available; run advise first")
    advice_dir = store.advice_dir(rev)
    proposal = _read_json(advice_dir / "proposal.json")
    if not isinstance(proposal, dict):
        raise SystemExit("proposal.json missing/invalid")
    options = proposal.get("options") or []
    if not isinstance(options, list) or not options:
        raise SystemExit("proposal.options empty")
    if args.option_index < 0 or args.option_index >= len(options):
        raise SystemExit("option_index out of range")
    opt = options[args.option_index]
    if not isinstance(opt, Mapping):
        raise SystemExit("selected option invalid")
    spec_delta = opt.get("spec_delta")
    cfg_delta = opt.get("cfg_delta")

    store.mark_advice_decision(rev=rev, decision="accepted", reason=args.reason)

    # Continue session with LLM as actor (trace will record it).
    source, _, _ = _load_benchmark(state.bench_id)
    strat = InteractiveSessionStrategy(measure_fn=_mock_measure_fn if args.mock else None)
    strat.continue_session(
        session_run_dir=args.session_run_dir,
        source=source,
        spec_delta=spec_delta if isinstance(spec_delta, dict) else None,
        cfg_delta=cfg_delta if isinstance(cfg_delta, dict) else None,
        next_phase=args.next_phase,
        actor="llm",
        reason=args.reason or f"accept option {args.option_index}",
    )
    print("accepted.")
    print("meta_report: " + str(Path(args.session_run_dir) / "session" / "meta_report.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive session runner (Step6/7).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="Create a new session and run phases.")
    p_new.add_argument("--bench", required=True, help="Benchmark name (rc, ota, opamp3).")
    p_new.add_argument("--out", type=Path, default=REPO_ROOT / "examples" / "output", help="Workspace root (runs/...).")
    p_new.add_argument("--seed", type=int, default=0)
    p_new.add_argument("--max-iters", type=int, default=21)
    p_new.add_argument("--run-to-phase", type=str, default="p2_corner_validate", choices=("p0_baseline", "p1_grid", "p2_corner_validate"))
    p_new.add_argument("--skip-baseline", action="store_true")
    p_new.add_argument("--reason", type=str, default=None)
    p_new.add_argument("--mock", action="store_true", help="Run with deterministic mock measure_fn (no ngspice).")

    # Grid config
    p_new.add_argument("--mode", type=str, default="coordinate", choices=("coordinate", "factorial"))
    p_new.add_argument("--levels", type=int, default=7)
    p_new.add_argument("--span-mul", type=float, default=4.0)
    p_new.add_argument("--scale", type=str, default="log", choices=("log", "linear"))
    p_new.add_argument("--continue-after-baseline-pass", action="store_true")

    # Corner validate config
    p_new.add_argument("--candidates-source", type=str, default="topk", choices=("topk", "pareto"))
    p_new.add_argument("--corners", type=str, default="oat", choices=("global", "oat", "oat_topm"))
    p_new.add_argument("--top-m", type=int, default=3)
    p_new.add_argument("--override-mode", type=str, default="add", choices=("set", "add", "mul"))
    p_new.add_argument("--no-clamp", action="store_true")
    p_new.set_defaults(fn=cmd_new)

    p_cont = sub.add_parser("continue", help="Continue an existing session with optional deltas.")
    p_cont.add_argument("--session-run-dir", type=Path, required=True)
    p_cont.add_argument("--next-phase", type=str, default="p2_corner_validate", choices=("p0_baseline", "p1_grid", "p2_corner_validate"))
    p_cont.add_argument("--spec-delta", type=str, default=None, help="JSON string or path to spec delta.")
    p_cont.add_argument("--cfg-delta", type=str, default=None, help="JSON string or path to cfg delta.")
    p_cont.add_argument("--reason", type=str, default=None)
    p_cont.add_argument("--mock", action="store_true", help="Run with deterministic mock measure_fn (no ngspice).")
    p_cont.set_defaults(fn=cmd_continue)

    p_inspect = sub.add_parser("inspect", help="Regenerate session meta report.")
    p_inspect.add_argument("--session-run-dir", type=Path, required=True)
    p_inspect.set_defaults(fn=cmd_inspect)

    p_advise = sub.add_parser("advise", help="Generate LLM insights + proposal for this session.")
    p_advise.add_argument("--session-run-dir", type=Path, required=True)
    p_advise.add_argument("--provider", type=str, default="openai", choices=("openai", "mock"))
    p_advise.add_argument("--model", type=str, default="gpt-4.1")
    p_advise.add_argument("--temperature", type=float, default=0.2)
    p_advise.add_argument("--max-tokens", type=int, default=None)
    p_advise.add_argument("--seed", type=int, default=None)
    p_advise.add_argument("--max-repairs", type=int, default=1)
    p_advise.add_argument("--mock-llm", action="store_true", help="Use built-in canned mock outputs.")
    p_advise.set_defaults(fn=cmd_advise)

    p_show = sub.add_parser("show-advice", help="Print proposal options for the latest advice.")
    p_show.add_argument("--session-run-dir", type=Path, required=True)
    p_show.add_argument("--advice-rev", type=int, default=None)
    p_show.set_defaults(fn=cmd_show_advice)

    p_reject = sub.add_parser("reject", help="Reject latest advice (record decision only).")
    p_reject.add_argument("--session-run-dir", type=Path, required=True)
    p_reject.add_argument("--advice-rev", type=int, default=None)
    p_reject.add_argument("--reason", type=str, default="")
    p_reject.set_defaults(fn=cmd_reject)

    p_accept = sub.add_parser("accept", help="Accept a proposal option and continue the session.")
    p_accept.add_argument("--session-run-dir", type=Path, required=True)
    p_accept.add_argument("--advice-rev", type=int, default=None)
    p_accept.add_argument("--option-index", type=int, required=True)
    p_accept.add_argument(
        "--next-phase",
        type=str,
        default="p2_corner_validate",
        choices=("p0_baseline", "p1_grid", "p2_corner_validate"),
    )
    p_accept.add_argument("--reason", type=str, default=None)
    p_accept.add_argument("--mock", action="store_true", help="Run with deterministic mock measure_fn (no ngspice).")
    p_accept.set_defaults(fn=cmd_accept)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
