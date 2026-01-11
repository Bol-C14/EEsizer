import shutil
from pathlib import Path

import pytest

from eesizer_core.contracts import SimPlan, SimRequest, CircuitSource, SourceKind
from eesizer_core.contracts.enums import SimKind
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim import DeckBuildOperator, NgspiceRunOperator
from eesizer_core.metrics import ComputeMetricsOperator


@pytest.mark.integration
def test_ngspice_ac_with_include(tmp_path):
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice not installed")

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    netlist_path = examples_dir / "rc_lowpass_include.sp"
    circuit_source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )
    plan = SimPlan(
        sims=(
            SimRequest(
                kind=SimKind.ac,
                params={"points_per_decade": 5, "start_hz": 10, "stop_hz": 1e6, "output_nodes": ["out"]},
            ),
        )
    )

    deck = DeckBuildOperator().run({"circuit_source": circuit_source, "sim_plan": plan, "sim_kind": SimKind.ac}, ctx=None).outputs[
        "deck"
    ]

    ctx = RunContext(workspace_root=tmp_path)
    runner = NgspiceRunOperator()
    result = runner.run({"deck": deck, "stage": "ac_integration"}, ctx)
    raw = result.outputs["raw_data"]

    metrics = ComputeMetricsOperator().run(
        {"raw_data": raw, "metric_names": ["ac_mag_db_at_1k", "ac_unity_gain_freq"]},
        ctx=None,
    ).outputs["metrics"]

    assert raw.outputs["ac_csv"].exists()
    assert metrics.values["ac_mag_db_at_1k"].value is not None
    # unity gain freq may be None if not crossed; ensure metric recorded
    assert "ac_unity_gain_freq" in metrics.values
