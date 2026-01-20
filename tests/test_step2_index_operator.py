import pytest

from eesizer_core.contracts.errors import ValidationError
from eesizer_core.operators.netlist import SpiceIndexOperator


def test_index_operator_rejects_non_string_netlist():
    op = SpiceIndexOperator()
    with pytest.raises(ValidationError, match="netlist_text"):
        op.run({"netlist_text": 123}, ctx=None)


def test_index_operator_rejects_control_block():
    netlist = ".control\nfoo\n.endc\nR1 in out 1k\n"
    op = SpiceIndexOperator()
    with pytest.raises(ValidationError, match="sanitize"):
        op.run({"netlist_text": netlist}, ctx=None)


def test_index_operator_rejects_bad_includes_type():
    op = SpiceIndexOperator()
    with pytest.raises(ValidationError, match="includes"):
        op.run({"netlist_text": "R1 in out 1k\n", "includes": "not-a-list"}, ctx=None)


def test_index_operator_accepts_includes_list():
    netlist = ".include foo.sp\nR1 in out 1k\n"
    op = SpiceIndexOperator()
    result = op.run({"netlist_text": netlist, "includes": ["foo.sp"]}, ctx=None)

    circuit_ir = result.outputs["circuit_ir"]
    assert circuit_ir.includes == ("foo.sp",)
