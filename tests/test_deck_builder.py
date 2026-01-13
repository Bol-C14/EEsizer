import pytest
from pathlib import Path

from eesizer_core.contracts import SimPlan, SimRequest, CircuitSource, SourceKind
from eesizer_core.contracts.errors import ValidationError
from eesizer_core.contracts.enums import SimKind
from eesizer_core.sim import DeckBuildOperator


def test_deck_builder_injects_ac_control_block():
    netlist = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end\n"
    plan = SimPlan(
        sims=(
            SimRequest(
                kind=SimKind.ac,
                params={
                    "points_per_decade": 5,
                    "start_hz": 10,
                    "stop_hz": 1e5,
                    "output_nodes": ["out"],
                },
            ),
        )
    )

    op = DeckBuildOperator()
    result = op.run({"netlist_text": netlist, "sim_plan": plan, "sim_kind": SimKind.ac}, ctx=None)
    deck = result.outputs["deck"]

    assert deck.kind == SimKind.ac
    assert deck.expected_outputs["ac_csv"] == "ac.csv"
    assert deck.expected_outputs_meta["ac_csv"] == ("frequency", "real(v(out))", "imag(v(out))")

    deck_text = deck.text
    assert ".control" in deck_text
    assert "ac dec 5 10 100000" in deck_text
    assert "wrdata __OUT__/ac.csv frequency real(v(out)) imag(v(out))" in deck_text
    assert deck_text.strip().endswith(".end")


def test_deck_builder_rejects_missing_ac_plan():
    netlist = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end\n"
    plan = SimPlan(sims=(SimRequest(kind=SimKind.tran, params={}),))
    op = DeckBuildOperator()

    with pytest.raises(ValidationError, match="SimKind 'ac'"):
        op.run({"netlist_text": netlist, "sim_plan": plan, "sim_kind": SimKind.ac}, ctx=None)


def test_deck_builder_dc_block():
    netlist = "V1 in 0 0\nR1 in out 1k\nC1 out 0 1u\n.end\n"
    plan = SimPlan(
        sims=(
            SimRequest(
                kind=SimKind.dc,
                params={"sweep_source": "V1", "sweep_node": "in", "start": 0, "stop": 1, "step": 0.5},
            ),
        )
    )
    op = DeckBuildOperator()
    deck = op.run({"netlist_text": netlist, "sim_plan": plan}, ctx=None).outputs["deck"]

    assert deck.kind == SimKind.dc
    deck_text = deck.text
    assert "dc V1 0 1 0.5" in deck_text
    assert "wrdata __OUT__/dc.csv v(in) v(out)" in deck_text
    assert deck.expected_outputs["dc_csv"] == "dc.csv"
    assert deck.expected_outputs_meta["dc_csv"] == ("v(in)", "v(out)")


def test_deck_builder_tran_block():
    netlist = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.end\n"
    plan = SimPlan(
        sims=(
            SimRequest(
                kind=SimKind.tran,
                params={"step": 1e-6, "stop": 1e-3, "output_nodes": ["out"]},
            ),
        )
    )
    op = DeckBuildOperator()
    deck = op.run({"netlist_text": netlist, "sim_plan": plan}, ctx=None).outputs["deck"]

    assert deck.kind == SimKind.tran
    deck_text = deck.text
    assert "tran 1e-06 0.001" in deck_text
    assert "wrdata __OUT__/tran.csv time v(out)" in deck_text
    assert deck.expected_outputs["tran_csv"] == "tran.csv"
    assert deck.expected_outputs_meta["tran_csv"] == ("time", "v(out)")


def test_deck_builder_preserves_base_dir():
    netlist = "V1 in 0 1\nR1 in out 1k\n.end\n"
    plan = SimPlan(sims=(SimRequest(kind=SimKind.ac, params={"output_nodes": ["out"]}),))
    base_dir = Path("/tmp/mycircuits")
    circuit_source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist,
        metadata={"base_dir": base_dir},
    )
    deck = DeckBuildOperator().run({"circuit_source": circuit_source, "sim_plan": plan}, ctx=None).outputs["deck"]
    assert deck.workdir == base_dir


def test_deck_builder_rejects_control_block():
    netlist = """
V1 in 0 1
.control
echo hello
.endc
.end
"""
    plan = SimPlan(sims=(SimRequest(kind=SimKind.ac, params={"output_nodes": ["out"]}),))
    op = DeckBuildOperator()
    with pytest.raises(ValidationError, match="must not contain .control"):
        op.run({"netlist_text": netlist, "sim_plan": plan}, ctx=None)
