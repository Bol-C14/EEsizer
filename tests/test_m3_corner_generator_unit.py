from eesizer_core.contracts import ParamDef, ParamSpace
from eesizer_core.search.corners import build_corner_set


def test_corner_generator_oat_count_and_ids():
    param_space = ParamSpace.build(
        [
            ParamDef(param_id="r1.value"),
            ParamDef(param_id="c1.value"),
        ]
    )
    nominal_values = {"r1.value": 1000.0, "c1.value": 1e-12}

    corner_set = build_corner_set(param_space=param_space, nominal_values=nominal_values, span_mul=10.0, mode="oat")
    corner_ids = [corner["corner_id"] for corner in corner_set["corners"]]

    assert len(corner_ids) == 7
    assert corner_ids == [
        "nominal",
        "all_low",
        "all_high",
        "c1.value_low",
        "c1.value_high",
        "r1.value_low",
        "r1.value_high",
    ]
