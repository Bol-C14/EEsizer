from eesizer_core.analysis.corners import aggregate_corner_results


def test_corner_aggregate_worst_score_and_losses():
    results = [
        {"corner_id": "nominal", "score": 1.0, "losses": [0.2, 0.5], "all_pass": True},
        {"corner_id": "fast", "score": 3.0, "losses": [0.1, 0.7], "all_pass": False},
        {"corner_id": "slow", "score": 2.0, "losses": [0.4, 0.4], "all_pass": True},
    ]
    summary = aggregate_corner_results(results)

    assert summary["worst_score"] == 3.0
    assert summary["worst_corner_id"] == "fast"
    assert summary["robust_losses"] == [0.4, 0.7]
    assert summary["pass_rate"] == 2 / 3
