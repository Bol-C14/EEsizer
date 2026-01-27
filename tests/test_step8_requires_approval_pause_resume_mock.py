from __future__ import annotations

from pathlib import Path

from eesizer_core.contracts import CircuitSource
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.runtime.session_plan_runner import actions_from_plan_option, execute_plan_option
from eesizer_core.runtime.session_plan_tools import build_session_plan_registry


def test_step8_requires_approval_pause_and_resume(tmp_path: Path) -> None:
    def dummy_run_grid_search(_inputs, _ctx, params):
        outs = params.get("_action_outputs") or []
        out = {}
        for name in outs:
            if name.endswith("/run_dir"):
                out[name] = "runs/p1"
            else:
                out[name] = []
        return out

    # Minimal plan dir + session dir layout.
    session_run_dir = tmp_path / "runs" / "sess1"
    plan_dir = session_run_dir / "session" / "llm" / "plan_advice" / "plan_rev0000"
    plan_dir.mkdir(parents=True, exist_ok=True)

    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n.end\n")

    plan_option = {
        "title": "A",
        "plan": [
            {
                "id": "a01",
                "op": "run_grid_search",
                "inputs": ["session_run_dir", "source"],
                "outputs": ["p1/run_dir", "p1/topk", "p1/pareto"],
                "params": {},
                "requires_approval": True,
            },
            {
                "id": "a02",
                "op": "run_grid_search",
                "inputs": ["session_run_dir", "source"],
                "outputs": ["p1_2/run_dir", "p1_2/topk", "p1_2/pareto"],
                "params": {},
                "requires_approval": False,
            },
        ],
    }

    registry = build_session_plan_registry(tool_fns={"run_grid_search": dummy_run_grid_search})
    actions = actions_from_plan_option(plan_option)

    # First run should pause before the first action (requires_approval).
    state1 = execute_plan_option(
        plan_dir=plan_dir,
        actions=actions,
        registry=registry,
        session_run_dir=session_run_dir,
        source=source,
        auto_approve=False,
        resume=False,
    )
    assert state1.status == "paused"
    assert state1.next_action_idx == 0

    # Resume with auto-approve should complete.
    state2 = execute_plan_option(
        plan_dir=plan_dir,
        actions=actions,
        registry=registry,
        session_run_dir=session_run_dir,
        source=source,
        auto_approve=True,
        resume=True,
    )
    assert state2.status == "completed"
