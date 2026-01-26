from eesizer_core.search.grid_config import parse_grid_search_config
from eesizer_core.search.ranges import RangeTrace, generate_candidates


def _make_ranges() -> list[RangeTrace]:
    return [
        RangeTrace(
            param_id="p1",
            nominal=2.0,
            lower=1.0,
            upper=3.0,
            scale="linear",
            levels=[1.0, 2.0, 3.0],
            source="bounds",
            span_mul=None,
            sanity={},
        ),
        RangeTrace(
            param_id="p2",
            nominal=5.0,
            lower=4.0,
            upper=6.0,
            scale="linear",
            levels=[4.0, 5.0, 6.0],
            source="bounds",
            span_mul=None,
            sanity={},
        ),
    ]


def test_coordinate_candidate_count():
    cfg = parse_grid_search_config({"grid_search": {"mode": "coordinate", "include_nominal": False}}, seed=0)
    ranges = _make_ranges()
    baseline = {"p1": 2.0, "p2": 5.0}

    candidates, meta = generate_candidates(ranges, baseline, cfg, max_candidates=None)

    assert len(candidates) == 4
    assert meta["total_generated"] == 4
    assert all(len(cand) == 1 for cand in candidates)


def test_factorial_candidate_count_and_truncation_deterministic():
    cfg = parse_grid_search_config(
        {
            "grid_search": {
                "mode": "factorial",
                "include_nominal": False,
                "truncate_policy": "seed_shuffle",
                "seed": 7,
            }
        },
        seed=7,
    )
    ranges = _make_ranges()
    baseline = {"p1": 2.0, "p2": 5.0}

    candidates_a, meta_a = generate_candidates(ranges, baseline, cfg, max_candidates=3)
    candidates_b, meta_b = generate_candidates(ranges, baseline, cfg, max_candidates=3)

    assert meta_a["total_generated"] == 8
    assert meta_a["truncated"] is True
    assert meta_a["truncated_to"] == 3
    assert candidates_a == candidates_b
    assert meta_a == meta_b
