from __future__ import annotations

from pathlib import Path
import os

import pytest

from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.session_llm_advice import SessionLLMAdviseOperator
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.session_store import SessionStore


@pytest.mark.integration
def test_step7_openai_integration_generates_valid_artifacts(tmp_path: Path) -> None:
    # This test is intentionally gated: no API key -> skip.
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    try:
        import openai  # noqa: F401
    except Exception:
        pytest.skip("openai package not installed")

    run_dir = tmp_path / "runs" / "sess1"
    store = SessionStore(run_dir)
    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3), seed=0, notes={})
    store.create_session(session_id="sess1", bench_id="ota", seed=0, spec=spec, cfg=cfg)
    (run_dir / "session" / "meta_report.md").write_text("# Session Meta Report\n", encoding="utf-8")

    grid_run_dir = tmp_path / "runs" / "grid1"
    (grid_run_dir / "search").mkdir(parents=True, exist_ok=True)
    (grid_run_dir / "report.md").write_text("## Run Summary\nok\n", encoding="utf-8")
    (grid_run_dir / "search" / "topk.json").write_text("[]", encoding="utf-8")
    (grid_run_dir / "search" / "pareto.json").write_text("[]", encoding="utf-8")
    store.write_checkpoint("p1_grid", {"phase_id": "p1_grid", "input_hash": "sha256:stub", "run_dir": str(grid_run_dir)})

    ctx = RunContext(workspace_root=tmp_path, run_id="sess1", seed=0)
    op = SessionLLMAdviseOperator()
    res = op.run(
        {
            "session_run_dir": run_dir,
            "llm_config": {"provider": "openai", "model": "gpt-4.1", "temperature": 0.0, "max_tokens": 800},
            "max_repairs": 1,
        },
        ctx=ctx,
    )

    advice_rel = res.outputs["advice_dir"]
    assert (run_dir / advice_rel / "insights.json").exists()
    assert (run_dir / advice_rel / "proposal.json").exists()

