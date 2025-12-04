from eesizer_core.netlist_patch import ParamChange, apply_param_changes


SIMPLE_NETLIST = """\
M1 out in 0 0 nch W=1u L=90n
R1 out 0 10k
"""


def test_apply_param_changes_scales_and_sets():
    changes = [
        ParamChange(component="M1", parameter="W", operation="scale", value=2.0),
        ParamChange(component="R1", parameter="R", operation="set", value=5000.0),
    ]
    patched, applied = apply_param_changes(SIMPLE_NETLIST, changes)
    assert "W=2u" in patched
    assert ("R=5000" in patched) or ("5k" in patched)
    assert applied["M1.W"] == pytest.approx(2e-6)
    assert applied["R1.R"] == 5000.0


def test_apply_param_changes_skips_missing():
    changes = [ParamChange(component="M1", parameter="X", operation="scale", value=2.0)]
    patched, applied = apply_param_changes(SIMPLE_NETLIST, changes)
    assert patched.strip().startswith("M1 out in 0 0 nch W=1u")
    assert applied == {}
import pytest
