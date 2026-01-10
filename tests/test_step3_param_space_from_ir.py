from eesizer_core.domain.spice.signature import topology_signature
from eesizer_core.domain.spice.params import ParamInferenceRules, infer_param_space_from_ir


def test_param_space_includes_mos_params():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    sig = topology_signature(netlist)
    cir = sig.circuit_ir
    ps = infer_param_space_from_ir(cir)
    ids = {p.param_id for p in ps.params}
    assert "m1.w" in ids
    assert "m1.l" in ids


def test_param_space_respects_frozen_ids():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir, frozen_param_ids=["m1.l"])
    pd = ps.get("m1.l")
    assert pd is not None and pd.frozen


def test_param_space_includes_passive_and_param_line():
    netlist = """
R1 in out 1k
.param W0=1u L0=0.1u
"""
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir)
    ids = {p.param_id for p in ps.params}
    assert "r1.value" in ids
    assert "param.w0" in ids


def test_param_space_includes_subckt_params():
    netlist = "XU1 in out opamp gain=10\n"
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir)
    ids = {p.param_id for p in ps.params}
    assert "xu1.gain" in ids


def test_param_space_allow_deny_rules():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    rules = ParamInferenceRules(deny_patterns=[r"^m1\.l$"])
    ps = infer_param_space_from_ir(cir, rules=rules)
    ids = {p.param_id for p in ps.params}
    assert "m1.l" not in ids
    assert "m1.w" in ids
