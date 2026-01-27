from __future__ import annotations

from pathlib import Path
import json

from eesizer_core.contracts import CircuitSource, CircuitSpec, MetricValue, MetricsBundle, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.session_llm_plan_advice import SessionLLMPlanAdviceOperator
from eesizer_core.runtime.session_plan_execution import build_session_plan_tool_fns
from eesizer_core.runtime.session_plan_runner import actions_from_plan_option, execute_plan_option
from eesizer_core.runtime.session_plan_tools import build_session_plan_registry
from eesizer_core.runtime.session_store import SessionStore
from eesizer_core.strategies.interactive_session import InteractiveSessionStrategy


def test_step8_execute_plan_end_to_end_mock(tmp_path: Path) -> None:
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
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "linear", "continue_after_baseline_pass": True},
            "corner_validate": {"candidates_source": "topk", "corners": "global"},
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

    plan_options = {
        "options": [
            {
                "title": "A",
                "intent": "mock",
                "plan": [
                    {
                        "id": "a01",
                        "op": "update_cfg",
                        "inputs": ["session_run_dir", "source"],
                        "outputs": ["session/cfg_rev", "session/cfg_hash"],
                        "params": {"cfg_delta": {"notes": {"grid_search": {"levels": 3}}}},
                        "requires_approval": False,
                        "notes": "",
                    },
                    {
                        "id": "a02",
                        "op": "run_grid_search",
                        "inputs": ["session_run_dir", "source"],
                        "outputs": ["p1/run_dir", "p1/topk", "p1/pareto"],
                        "params": {},
                        "requires_approval": False,
                        "notes": "",
                    },
                    {
                        "id": "a03",
                        "op": "run_corner_validate",
                        "inputs": ["session_run_dir", "source"],
                        "outputs": ["p2/run_dir", "p2/robust_topk", "p2/robust_pareto"],
                        "params": {},
                        "requires_approval": False,
                        "notes": "",
                    },
                ],
            }
        ]
    }

    op = SessionLLMPlanAdviceOperator()
    op.run(
        {
            "session_run_dir": session_ctx.run_dir(),
            "llm_config": {"provider": "mock", "model": "mock", "temperature": 0.0},
            "mock_response": json.dumps(plan_options),
        },
        ctx=session_ctx,
    )

    plan_dir = store.plan_dir(0)
    assert (plan_dir / "plan_options.json").exists()

    tool_fns = build_session_plan_tool_fns(measure_fn=measure_fn)
    registry = build_session_plan_registry(tool_fns=tool_fns)
    actions = actions_from_plan_option(plan_options["options"][0])

    state_out = execute_plan_option(
        plan_dir=plan_dir,
        actions=actions,
        registry=registry,
        session_run_dir=session_ctx.run_dir(),
        source=source,
        auto_approve=True,
        resume=False,
    )

    assert state_out.status == "completed"
    assert (plan_dir / "execution.jsonl").exists()
    assert (plan_dir / "execution_state.json").exists()
    assert store.load_session_state().cfg_rev == 1

