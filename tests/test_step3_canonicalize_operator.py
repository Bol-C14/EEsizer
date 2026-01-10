from eesizer_core.operators.netlist import SpiceCanonicalizeOperator


def test_spice_canonicalize_operator_pipeline():
    netlist = """
M1 d g s b nmos W=1u L=0.1u
.control
echo foo
.endc
"""
    op = SpiceCanonicalizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)
    sanitized = result.outputs["sanitized_text"]
    cir = result.outputs["circuit_ir"]
    sig = result.outputs["topology_signature"]

    assert ".control" not in sanitized
    assert "m1.w" in cir.param_locs
    assert sig
