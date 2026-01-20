import pytest

from eesizer_core.contracts.errors import ValidationError
from eesizer_core.operators.netlist import TopologySignatureOperator


def test_signature_operator_ignores_include_paths_when_disabled():
    netlist_a = ".include foo.sp\nR1 in out 1k\n"
    netlist_b = ".include bar.sp\nR1 in out 1k\n"
    op = TopologySignatureOperator()

    sig_a = op.run({"netlist_text": netlist_a, "include_paths": False}, ctx=None).outputs["signature"]
    sig_b = op.run({"netlist_text": netlist_b, "include_paths": False}, ctx=None).outputs["signature"]

    assert sig_a == sig_b

    sig_a_inc = op.run({"netlist_text": netlist_a, "include_paths": True}, ctx=None).outputs["signature"]
    sig_b_inc = op.run({"netlist_text": netlist_b, "include_paths": True}, ctx=None).outputs["signature"]
    assert sig_a_inc != sig_b_inc


def test_signature_operator_rejects_invalid_include_paths():
    op = TopologySignatureOperator()
    with pytest.raises(ValidationError, match="include_paths"):
        op.run({"netlist_text": "R1 in out 1k\n", "include_paths": "nope"}, ctx=None)
