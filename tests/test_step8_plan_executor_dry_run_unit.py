from __future__ import annotations

from pathlib import Path

from eesizer_core.runtime.session_plan_tools import build_session_plan_registry
from eesizer_core.runtime.session_plan_runner import write_dry_run
from eesizer_core.runtime.tool_catalog import build_tool_catalog


def test_step8_dry_run_writes_summary(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan_rev0000"
    plan_dir.mkdir(parents=True, exist_ok=True)

    reg = build_session_plan_registry()
    catalog = build_tool_catalog(reg)

    option = {
        "title": "A",
        "plan": [
            {
                "id": "a01",
                "op": "run_grid_search",
                "inputs": ["session_run_dir", "source"],
                "outputs": ["p1/run_dir", "p1/topk", "p1/pareto"],
                "params": {},
                "requires_approval": False,
            }
        ],
    }

    out = write_dry_run(plan_dir=plan_dir, plan_option=option, tool_catalog=catalog)
    text = out.read_text(encoding="utf-8")
    assert "Plan Dry Run" in text
    assert "## Actions" in text
    assert "run_grid_search" in text

