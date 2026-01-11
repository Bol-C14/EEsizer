import pytest

from eesizer_core.contracts.artifacts import Patch, PatchOp
from eesizer_core.contracts.enums import PatchOpType
from eesizer_core.domain.spice.params import infer_param_space_from_ir
from eesizer_core.domain.spice.patching import validate_patch
from eesizer_core.domain.spice import topology_signature


def test_wl_ratio_check_handles_hierarchical_ids():
    netlist = "X1 n1 n2 n3 n4 myblock\n.subckt myblock d g s b\nM1 d g s b nmos W=1u L=0.1u\n.ends\n"
    cir = topology_signature(netlist).circuit_ir
    ps = infer_param_space_from_ir(cir)

    patch = Patch(
        ops=(
            PatchOp(param="x1.m1.w", op=PatchOpType.set, value="0.05u"),
            PatchOp(param="x1.m1.l", op=PatchOpType.set, value="0.1u"),
        )
    )

    result = validate_patch(cir=cir, param_space=ps, patch=patch, wl_ratio_min=0.5)
    assert not result.ok
    assert any("x1.m1.w" in e for e in result.errors + result.ratio_errors)
