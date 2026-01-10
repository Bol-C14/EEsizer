from eesizer_core.domain.spice.signature import topology_signature


def test_parse_elements_and_params():
    netlist = """
M1 d g s b nmos W=1u L=0.1u
R1 d out 1k
"""
    result = topology_signature(netlist)
    assert result.signature
    assert "M1" in result.circuit_ir.elements
    assert "R1" in result.circuit_ir.elements
    assert "m1.w" in result.circuit_ir.param_locs
    assert "m1.l" in result.circuit_ir.param_locs


def test_signature_invariant_on_value_change():
    netlist_a = "M1 d g s b nmos W=1u L=0.1u\n"
    netlist_b = "M1 d g s b nmos W=2u L=0.1u\n"
    sig_a = topology_signature(netlist_a).signature
    sig_b = topology_signature(netlist_b).signature
    assert sig_a == sig_b


def test_signature_changes_on_topology_change():
    netlist_base = "M1 d g s b nmos W=1u L=0.1u\n"
    netlist_changed = "M1 d g x b nmos W=1u L=0.1u\n"  # source node changed
    sig_base = topology_signature(netlist_base).signature
    sig_changed = topology_signature(netlist_changed).signature
    assert sig_base != sig_changed


def test_control_block_stripped():
    netlist_with_control = """
M1 d g s b nmos W=1u L=0.1u
.control
echo foo
.endc
"""
    netlist_plain = "M1 d g s b nmos W=1u L=0.1u\n"
    sig_ctrl = topology_signature(netlist_with_control).signature
    sig_plain = topology_signature(netlist_plain).signature
    assert sig_ctrl == sig_plain
