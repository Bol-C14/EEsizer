from eesizer_core.contracts import CircuitSource, SourceKind
from eesizer_core.domain.spice import topology_signature
from eesizer_core.domain.spice.params import infer_param_space_from_ir
from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.operators.netlist import PatchApplyOperator


def test_patch_apply_operator_end_to_end():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    src = CircuitSource(kind=SourceKind.spice_netlist, text=netlist, name="test")

    sig = topology_signature(netlist)
    cir = sig.circuit_ir
    ps = infer_param_space_from_ir(cir)

    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))

    op = PatchApplyOperator()
    result = op.run({"source": src, "param_space": ps, "patch": patch}, ctx=None)

    new_src = result.outputs["source"]
    new_sig = result.outputs["topology_signature"]

    assert "2u" in new_src.text
    assert sig.signature == new_sig
