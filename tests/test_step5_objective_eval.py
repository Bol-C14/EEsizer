from eesizer_core.contracts import CircuitSpec, Objective, MetricValue, MetricsBundle
from eesizer_core.strategies.objective_eval import evaluate_objectives


def _bundle(values: dict[str, float | None]) -> MetricsBundle:
    mb = MetricsBundle()
    for k, v in values.items():
        mb.values[k] = MetricValue(name=k, value=v, unit="")
    return mb


def test_objective_eval_handles_ge_and_le():
    spec = CircuitSpec(
        objectives=(
            Objective(metric="gain", target=10.0, sense="ge"),
            Objective(metric="noise", target=1.0, sense="le"),
        )
    )
    metrics = _bundle({"gain": 12.0, "noise": 0.5})
    result = evaluate_objectives(spec, metrics)
    assert result["all_pass"] is True
    assert result["score"] == 0.0


def test_objective_eval_eq_with_tol_and_missing_value_penalty():
    spec = CircuitSpec(
        objectives=(
            Objective(metric="offset", target=0.0, sense="eq", tol=0.01, weight=2.0),
            Objective(metric="missing", target=1.0, sense="ge"),
        )
    )
    metrics = _bundle({"offset": 0.02})
    result = evaluate_objectives(spec, metrics)
    per = result["per_objective"]
    # offset penalty: (0.02 - 0.01) / (abs(0)+eps) -> finite positive
    assert per[0]["passed"] is False
    assert per[1]["passed"] is False
    assert result["all_pass"] is False
    assert result["score"] > 0.0


def test_objective_eval_score_improves_when_closer_to_target():
    spec = CircuitSpec(objectives=(Objective(metric="bw", target=1.0e6, sense="ge"),))
    metrics_far = _bundle({"bw": 0.5e6})
    metrics_close = _bundle({"bw": 0.9e6})
    score_far = evaluate_objectives(spec, metrics_far)["score"]
    score_close = evaluate_objectives(spec, metrics_close)["score"]
    assert score_close < score_far
