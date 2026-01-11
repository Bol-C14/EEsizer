import pytest

from eesizer_core.contracts import SimPlan, SimRequest
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
    result = op.run({"netlist_text": netlist, "sim_plan": plan}, ctx=None)
    deck = result.outputs["deck"]

    assert deck.kind == SimKind.ac
    assert deck.expected_outputs["ac_csv"] == "ac.csv"

    deck_text = deck.text
    assert ".control" in deck_text
    assert "ac dec 5 10 100000" in deck_text
    assert "wrdata ac.csv frequency vdb(out) vp(out)" in deck_text
    assert deck_text.strip().endswith(".end")


def test_deck_builder_rejects_missing_ac_plan():
    netlist = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end\n"
    plan = SimPlan(sims=(SimRequest(kind=SimKind.tran, params={}),))
    op = DeckBuildOperator()

    with pytest.raises(ValidationError, match="AC SimRequest"):
        op.run({"netlist_text": netlist, "sim_plan": plan}, ctx=None)
