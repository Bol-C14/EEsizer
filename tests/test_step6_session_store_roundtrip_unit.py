from pathlib import Path

from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.session_store import SessionStore


def test_session_store_create_and_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sess1"
    store = SessionStore(run_dir)

    spec = CircuitSpec(objectives=(Objective(metric="ugbw_hz", target=1e6, sense="ge"),))
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3), seed=0, notes={})

    state0 = store.create_session(session_id="sess1", bench_id="rc", seed=0, spec=spec, cfg=cfg)
    state1 = store.load_session_state()
    assert state0 == state1

    assert (run_dir / "session" / "session_state.json").exists()
    assert (run_dir / "session" / "spec_trace.jsonl").exists()
    assert (run_dir / "session" / "cfg_trace.jsonl").exists()
    assert (run_dir / "session" / "spec_revs" / "spec_rev0000.json").exists()
    assert (run_dir / "session" / "cfg_revs" / "cfg_rev0000.json").exists()

    ck = store.write_checkpoint("p1_grid", {"phase_id": "p1_grid", "input_hash": "sha256:x"})
    assert ck.exists()
    loaded = store.load_checkpoint("p1_grid")
    assert loaded and loaded["input_hash"] == "sha256:x"

