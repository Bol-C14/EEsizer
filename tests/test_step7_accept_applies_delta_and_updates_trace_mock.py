from __future__ import annotations

from pathlib import Path
import json

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.session_llm_advice import SessionLLMAdviseOperator
from eesizer_core.runtime.session_store import SessionStore
from eesizer_core.strategies.interactive_session import InteractiveSessionStrategy


def test_step7_accept_applies_delta_and_updates_trace(tmp_path: Path) -> None:
    calls = {"n": 0}

    def measure_fn(_src: CircuitSource, iter_idx: int) -> MetricsBundle:
        calls["n"] += 1
        mb = MetricsBundle()
        mb.values["ugbw_hz"] = MetricValue(name="ugbw_hz", value=1e6 + 1e3 * iter_idx, unit="Hz")
        mb.values["phase_margin_deg"] = MetricValue(name="phase_margin_deg", value=70.0, unit="deg")
        mb.values["power_w"] = MetricValue(name="power_w", value=1e-3, unit="W")
        return mb

    ws = tmp_path / "ws"
    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n.end\n")
    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=3, no_improve_patience=1),
        seed=0,
        notes={
            "session": {"run_baseline": False},
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear"},
        },
    )

    strat = InteractiveSessionStrategy(measure_fn=measure_fn)
    session_ctx = strat.start_session(
        bench_id="ota",
        source=source,
        spec=spec,
        cfg=cfg,
        workspace_root=ws,
        run_to_phase="p1_grid",
        reason="unit_test",
    )

    store = SessionStore(session_ctx.run_dir())
    ck_before = store.load_checkpoint("p1_grid") or {}
    old_grid_run_dir = ck_before.get("run_dir")
    assert old_grid_run_dir

    insights = json.dumps(
        {
            "summary": "ok",
            "tradeoffs": [],
            "sensitivity_rank": [],
            "robustness_notes": [],
            "recommended_actions": [],
        }
    )
    proposal = json.dumps(
        {
            "options": [
                {
                    "title": "Increase levels",
                    "intent": "explore more candidates",
                    "spec_delta": None,
                    "cfg_delta": {"notes": {"grid_search": {"levels": 3}}},
                    "plan": None,
                    "expected_effects": ["more samples"],
                    "risks": ["more budget"],
                    "budget_estimate": {"iters": 3, "corners": 0},
                }
            ]
        }
    )

    advise_op = SessionLLMAdviseOperator()
    advise_op.run(
        {
            "session_run_dir": session_ctx.run_dir(),
            "llm_config": {"provider": "mock", "model": "mock", "temperature": 0.0},
            "mock_response": insights,
            "mock_plan": proposal,
        },
        ctx=session_ctx,
    )

    state = store.load_session_state()
    assert state.latest_advice_rev == 0

    advice_dir = store.advice_dir(0)
    payload = json.loads((advice_dir / "proposal.json").read_text(encoding="utf-8"))
    opt0 = payload["options"][0]

    store.mark_advice_decision(rev=0, decision="accepted", reason="unit_test_accept")

    strat.continue_session(
        session_run_dir=session_ctx.run_dir(),
        source=source,
        cfg_delta=opt0.get("cfg_delta"),
        next_phase="p1_grid",
        actor="llm",
        reason="unit_test_accept",
    )

    state2 = store.load_session_state()
    assert state2.cfg_rev == 1

    ck_after = store.load_checkpoint("p1_grid") or {}
    assert ck_after.get("run_dir") != old_grid_run_dir

    cfg_trace = (session_ctx.run_dir() / "session" / "cfg_trace.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in cfg_trace if line.strip()]
    assert any(e.get("actor") == "llm" for e in entries)

