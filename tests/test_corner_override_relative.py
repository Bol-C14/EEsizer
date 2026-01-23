from eesizer_core.contracts import ParamDef, ParamSpace
from eesizer_core.search.corners import build_corner_set
from eesizer_core.strategies.corner_search.measurement import resolve_corner_overrides


def test_corner_override_add_is_relative_to_candidate():
    param_space = ParamSpace.build([ParamDef(param_id="r1.value", lower=100.0, upper=10000.0)])
    nominal_values = {"r1.value": 1000.0}
    corner_set = build_corner_set(
        param_space=param_space,
        nominal_values=nominal_values,
        span_mul=10.0,
        corner_param_ids=["r1.value"],
        include_global_corners=False,
        override_mode="add",
        mode="oat",
    )
    corner = [c for c in corner_set["corners"] if c["corner_id"] == "r1.value_low"][0]
    base_values = {"r1.value": 2000.0}

    applied, errors, warnings = resolve_corner_overrides(
        base_values=base_values,
        param_bounds=corner_set["param_bounds"],
        overrides=corner["overrides"],
        clamp=True,
    )

    assert not errors
    assert not warnings
    assert applied["r1.value"] == 1100.0
