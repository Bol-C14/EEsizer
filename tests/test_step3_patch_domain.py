from eesizer_core.contracts.artifacts import ParamDef, ParamSpace, Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.domain.spice.patching import (
    apply_patch_with_topology_guard,
    validate_patch,
)
from eesizer_core.domain.spice.signature import topology_signature


def test_apply_patch_changes_value_but_preserves_topology():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    sig = topology_signature(netlist)
    cir = sig.circuit_ir
    patch = Patch(ops=(PatchOp(param="M1.w", op=PatchOpType.set, value="2u"),))
    new_cir = apply_patch_with_topology_guard(cir, patch)
    assert "2u" in "\n".join(new_cir.lines)
    sig2 = topology_signature("\n".join(new_cir.lines))
    assert sig.signature == sig2.signature


def test_validate_patch_rejects_unknown_param():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="M1.w")])
    patch = Patch(ops=(PatchOp(param="M2.w", op=PatchOpType.set, value="2u"),))
    validation = validate_patch(cir, ps, patch)
    assert not validation.ok
    assert validation.errors


def test_validate_patch_rejects_frozen_param():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="M1.w", frozen=True)])
    patch = Patch(ops=(PatchOp(param="M1.w", op=PatchOpType.set, value="2u"),))
    validation = validate_patch(cir, ps, patch)
    assert not validation.ok
    assert validation.errors


def test_apply_patch_mul_numeric():
    netlist = "M1 d g s b nmos W=1e-6 L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="M1.w", op=PatchOpType.mul, value=2.0),))
    new_cir = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(new_cir.lines)
    assert "2e-06" in text or "2e-6" in text
