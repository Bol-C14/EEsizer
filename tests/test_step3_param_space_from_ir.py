from eesizer_core.domain.spice.signature import topology_signature
from eesizer_core.domain.spice.params import infer_param_space_from_ir


def test_param_space_includes_mos_params():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    sig = topology_signature(netlist)
    cir = sig.circuit_ir
    ps = infer_param_space_from_ir(cir)
    ids = {p.param_id for p in ps.params}
    assert "M1.w" in ids
    assert "M1.l" in ids


def test_param_space_respects_frozen_ids():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir, frozen_param_ids=["M1.l"])
    pd = ps.get("M1.l")
    assert pd is not None and pd.frozen
