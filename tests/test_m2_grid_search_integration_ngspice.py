from pathlib import Path

import pytest

from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import StrategyConfig, OptimizationBudget
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim.ngspice_runner import resolve_ngspice_executable
from eesizer_core.strategies import GridSearchStrategy


@pytest.mark.integration
def test_grid_search_integration_with_ngspice(tmp_path):
    if resolve_ngspice_executable() is None:
        pytest.skip("ngspice not installed")

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    netlist_path = examples_dir / "circuits" / "rc_lowpass.sp"
    source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))

    strategy = GridSearchStrategy()
    cfg = StrategyConfig(
        budget=OptimizationBudget(max_iterations=5, no_improve_patience=1),
        notes={
            "param_rules": {"allow_patterns": [r"^(r1\.value|c1\.value)$"]},
            "grid_search": {"mode": "coordinate", "levels": 2, "span_mul": 2.0, "scale": "log"},
        },
    )
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    assert result.best_metrics.values.get("ac_mag_db_at_1k") is not None
    assert result.history
    assert result.notes.get("best_score") <= result.history[0]["score"]
