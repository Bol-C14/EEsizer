import pytest

from eesizer_core.contracts.artifacts import ParamDef, ParamSpace
from eesizer_core.search.grid_config import parse_grid_search_config
from eesizer_core.search.ranges import infer_ranges


def test_infer_ranges_bounds_span_and_log_fallback():
    param_space = ParamSpace.build(
        [
            ParamDef(param_id="m1.w", lower=1e-6, upper=2e-6),
            ParamDef(param_id="m2.w"),
            ParamDef(param_id="r1.value", lower=-1.0, upper=1.0),
            ParamDef(param_id="c1.value"),
        ]
    )
    nominal_values = {"m1.w": 1.5e-6, "m2.w": 2e-6, "r1.value": -0.5}

    cfg = parse_grid_search_config(
        {
            "grid_search": {
                "levels": 3,
                "span_mul": 8.0,
                "scale": "log",
                "include_nominal": False,
            }
        },
        seed=0,
    )

    ranges = infer_ranges(["m1.w", "m2.w", "r1.value", "c1.value"], param_space, nominal_values, cfg)
    trace = {item.param_id: item for item in ranges}

    m1 = trace["m1.w"]
    assert m1.source == "bounds"
    assert m1.lower == pytest.approx(1e-6)
    assert m1.upper == pytest.approx(2e-6)
    assert m1.scale == "log"
    assert len(m1.levels) == 3

    m2 = trace["m2.w"]
    assert m2.source == "nominal*span_mul"
    assert m2.lower == pytest.approx(2e-6 / 8.0)
    assert m2.upper == pytest.approx(2e-6 * 8.0)

    r1 = trace["r1.value"]
    assert r1.scale == "linear"
    assert "log_fallback_linear" in r1.sanity.get("warnings", [])

    c1 = trace["c1.value"]
    assert c1.skipped is True
    assert c1.skip_reason == "missing_nominal"
