from eesizer_core.contracts.artifacts import CircuitSpec, Constraint, Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.domain.spice import topology_signature
from eesizer_core.domain.spice.params import infer_param_space_from_ir
from eesizer_core.operators.guards import PatchGuardOperator


def test_spec_constraints_ratio_min_rejected_and_accepted():
    netlist = "M1 d g s b nmos W=1u L=0.1u\nM2 d g s b nmos W=1u L=0.1u\n"
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir)
    spec = CircuitSpec(
        constraints=(
            Constraint(
                kind="param_ratio_min",
                data={"lhs": "m1.w", "rhs": "m2.w", "min_ratio": 2.0},
            ),
        )
    )

    op = PatchGuardOperator()
    bad_patch = Patch(
        ops=(
            PatchOp(param="m1.w", op=PatchOpType.set, value="1u"),
            PatchOp(param="m2.w", op=PatchOpType.set, value="1u"),
        )
    )
    bad_check = op.run(
        {"circuit_ir": cir, "param_space": ps, "patch": bad_patch, "spec": spec},
        ctx=None,
    ).outputs["check"]
    assert not bad_check.ok

    ok_patch = Patch(
        ops=(
            PatchOp(param="m1.w", op=PatchOpType.set, value="3u"),
            PatchOp(param="m2.w", op=PatchOpType.set, value="1u"),
        )
    )
    ok_check = op.run(
        {"circuit_ir": cir, "param_space": ps, "patch": ok_patch, "spec": spec},
        ctx=None,
    ).outputs["check"]
    assert ok_check.ok
