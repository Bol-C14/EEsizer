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
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    new_cir = apply_patch_with_topology_guard(cir, patch)
    assert "2u" in "\n".join(new_cir.lines)
    sig2 = topology_signature("\n".join(new_cir.lines))
    assert sig.signature == sig2.signature


def test_validate_patch_rejects_unknown_param():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w")])
    patch = Patch(ops=(PatchOp(param="m2.w", op=PatchOpType.set, value="2u"),))
    validation = validate_patch(cir, ps, patch)
    assert not validation.ok
    assert validation.errors


def test_validate_patch_rejects_frozen_param():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w", frozen=True)])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    validation = validate_patch(cir, ps, patch)
    assert not validation.ok
    assert validation.errors


def test_apply_patch_mul_numeric():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=2.0),))
    new_cir = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(new_cir.lines)
    assert "2e-06" in text or "2e-6" in text


def test_apply_patch_add_with_units():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.add, value="1u"),))
    new_cir = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(new_cir.lines)
    assert "2e-06" in text or "2e-6" in text


def test_apply_patch_repeated_value_length_change():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch1 = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="10u"),))
    patch2 = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    cir1 = apply_patch_with_topology_guard(cir, patch1)
    cir2 = apply_patch_with_topology_guard(cir1, patch2)
    text = "\n".join(cir2.lines)
    assert "W=2u" in text
    assert "2uu" not in text


def test_apply_patch_on_passive_main_value():
    netlist = "R1 in out 1k\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="R1.value", op=PatchOpType.mul, value=2),))
    cir2 = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(cir2.lines)
    assert "2e+03" in text or "2000" in text or "2e3" in text


def test_apply_patch_inline_comment_attached():
    netlist = "R1 in out 1k;comment\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="R1.value", op=PatchOpType.mul, value=2),))
    cir2 = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(cir2.lines)
    assert ";comment" in text
    assert "2e+03" in text or "2000" in text or "2000.0" in text or "2e3" in text


def test_apply_patch_on_param_line():
    netlist = ".param W0=1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="param.w0", op=PatchOpType.set, value="2u"),))
    cir2 = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(cir2.lines)
    assert "W0=2u" in text or "w0=2u" in text


def test_apply_patch_continuation_line():
    netlist = "M1 d g s b nmos W=1u\n+ L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    patch = Patch(ops=(PatchOp(param="m1.l", op=PatchOpType.set, value="0.2u"),))
    cir2 = apply_patch_with_topology_guard(cir, patch)
    text = "\n".join(cir2.lines)
    assert "L=0.2u" in text


def test_validate_patch_bounds():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w", lower=0.0, upper=2e-6)])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="3u"),))
    result = validate_patch(cir, ps, patch)
    assert not result.ok
    assert any("above upper" in e for e in result.errors)


def test_validate_patch_rejects_illegal_op():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w")])
    patch = Patch(ops=(PatchOp(param="m1.w", op="divide", value="2"),))  # type: ignore[arg-type]
    result = validate_patch(cir, ps, patch)
    assert not result.ok
    assert any("unsupported op" in e for e in result.errors)


def test_validate_patch_rejects_non_numeric_value():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w")])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="abc"),))
    result = validate_patch(cir, ps, patch)
    assert not result.ok
    assert any("non-numeric" in e for e in result.errors)


def test_validate_patch_rejects_large_mul_factor():
    netlist = "M1 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build([ParamDef(param_id="m1.w")])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=20.0),))
    result = validate_patch(cir, ps, patch)
    assert not result.ok
    assert any("exceeds max" in e for e in result.errors)
