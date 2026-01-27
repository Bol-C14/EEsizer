from __future__ import annotations

from pathlib import Path
import json

from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.operators.llm_context import build_session_llm_context
from eesizer_core.runtime.session_store import SessionStore


def test_step7_context_builder_collects_grid_artifacts(tmp_path: Path) -> None:
    # Create a minimal session with rev snapshots.
    run_dir = tmp_path / "runs" / "sess1"
    store = SessionStore(run_dir)
    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3), seed=0, notes={})
    store.create_session(session_id="sess1", bench_id="ota", seed=0, spec=spec, cfg=cfg)

    (run_dir / "session" / "meta_report.md").write_text("# Session Meta Report\n", encoding="utf-8")

    # Stub a grid run directory with the artifacts Step7 expects.
    grid_run_dir = tmp_path / "runs" / "grid1"
    (grid_run_dir / "search").mkdir(parents=True, exist_ok=True)
    (grid_run_dir / "insights").mkdir(parents=True, exist_ok=True)

    (grid_run_dir / "report.md").write_text(
        "\n".join(
            [
                "# Report",
                "",
                "## Run Summary",
                "ok",
                "",
                "## Top-K Candidates",
                "(stub)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (grid_run_dir / "search" / "topk.json").write_text(json.dumps([{"iteration": 0}], indent=2), encoding="utf-8")
    (grid_run_dir / "search" / "pareto.json").write_text(json.dumps([{"iteration": 0}], indent=2), encoding="utf-8")
    (grid_run_dir / "search" / "robust_topk.json").write_text(json.dumps([{"iteration": 0}], indent=2), encoding="utf-8")
    (grid_run_dir / "search" / "robust_pareto.json").write_text(json.dumps([{"iteration": 0}], indent=2), encoding="utf-8")
    (grid_run_dir / "insights" / "sensitivity.json").write_text(json.dumps({"rank": []}, indent=2), encoding="utf-8")

    store.write_checkpoint(
        "p1_grid",
        {"phase_id": "p1_grid", "input_hash": "sha256:stub", "run_dir": str(grid_run_dir)},
    )

    ctx = build_session_llm_context(store, max_topk=1, max_pareto=1, max_report_section_lines=20)

    assert ctx["session"]["bench_id"] == "ota"
    assert ctx["grid_run_dir"] == str(grid_run_dir)
    assert len(ctx["topk"]) == 1
    assert len(ctx["pareto"]) == 1
    assert "## Run Summary" in ctx["grid_report_sections"]

    evidence = ctx["evidence"]
    assert "session/meta_report.md" in evidence
    assert "session/spec_revs/spec_rev0000.json" in evidence
    assert "session/cfg_revs/cfg_rev0000.json" in evidence

