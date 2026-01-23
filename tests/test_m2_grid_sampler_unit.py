from eesizer_core.search.samplers import coordinate_candidates, make_levels


def test_make_levels_linear_sorted():
    levels = make_levels(nominal=10.0, lower=5.0, upper=20.0, levels=4, span_mul=10.0, scale="linear")
    assert levels == [5.0, 10.0, 15.0, 20.0]


def test_coordinate_candidates_skip_baseline():
    param_ids = ["p1", "p2"]
    per_param_levels = {"p1": [1.0, 2.0], "p2": [10.0, 20.0]}
    baseline_values = {"p1": 1.0, "p2": 10.0}
    candidates = coordinate_candidates(param_ids, per_param_levels, baseline_values)
    assert candidates == [{"p1": 2.0}, {"p2": 20.0}]
