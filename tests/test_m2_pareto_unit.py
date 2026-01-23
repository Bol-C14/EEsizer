from eesizer_core.analysis.pareto import objective_losses, pareto_front


def test_objective_losses():
    eval_dict = {
        "per_objective": [
            {"penalty": 0.5, "weight": 2.0},
            {"penalty": 1.0, "weight": 1.5},
        ]
    }
    losses = objective_losses(eval_dict)
    assert losses == [1.0, 1.5]


def test_pareto_front_indices():
    vectors = [
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [0.5, 3.0],
    ]
    front = pareto_front(vectors)
    assert sorted(front) == [0, 3]
