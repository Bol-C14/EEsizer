from eesizer_core.contracts.artifacts import ParamDef, ParamSpace, Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.domain.spice import topology_signature
from eesizer_core.operators.guards import PatchGuardOperator


def _make_ir_and_param_space():
    netlist = "M1 d g s b nmos W=1u L=0.1u\nR1 in out 1k\n"
    cir = topology_signature(netlist).circuit_ir
    ps = ParamSpace.build(
        [
            ParamDef(param_id="m1.w"),
            ParamDef(param_id="m1.l"),
            ParamDef(param_id="r1.value"),
        ]
    )
    return cir, ps


def test_patch_guard_rejects_frozen_param():
    cir, _ = _make_ir_and_param_space()
    ps = ParamSpace.build([ParamDef(param_id="m1.w", frozen=True)])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.set, value="2u"),))
    op = PatchGuardOperator()
    check = op.run({"circuit_ir": cir, "param_space": ps, "patch": patch}, ctx=None).outputs["check"]
    assert not check.ok
    assert any("frozen" in reason for reason in check.reasons)


def test_patch_guard_rejects_unknown_param():
    cir, ps = _make_ir_and_param_space()
    patch = Patch(ops=(PatchOp(param="m2.w", op=PatchOpType.set, value="2u"),))
    op = PatchGuardOperator()
    check = op.run({"circuit_ir": cir, "param_space": ps, "patch": patch}, ctx=None).outputs["check"]
    assert not check.ok
    assert any("unknown param" in reason for reason in check.reasons)


def test_patch_guard_rejects_large_mul_factor():
    cir, ps = _make_ir_and_param_space()
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=5.0),))
    op = PatchGuardOperator()
    check = op.run(
        {"circuit_ir": cir, "param_space": ps, "patch": patch, "guard_cfg": {"max_mul_factor": 2.0}},
        ctx=None,
    ).outputs["check"]
    assert not check.ok
    assert any("exceeds max" in reason for reason in check.reasons)


def test_patch_guard_rejects_negative_passive_value():
    cir, ps = _make_ir_and_param_space()
    patch = Patch(ops=(PatchOp(param="r1.value", op=PatchOpType.set, value=-1.0),))
    op = PatchGuardOperator()
    check = op.run({"circuit_ir": cir, "param_space": ps, "patch": patch}, ctx=None).outputs["check"]
    assert not check.ok
    assert any("non-negative" in reason for reason in check.reasons)
