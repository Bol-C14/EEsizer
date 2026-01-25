from eesizer_core.contracts import CircuitSpec, Objective, MetricsBundle, MetricValue
from eesizer_core.analysis.objective_eval import evaluate_objectives


def test_objective_eval_uses_alias_metric_name():
    spec = CircuitSpec(objectives=(Objective(metric="ac_unity_gain_freq", target=1.0, sense="ge"),))
    metrics = MetricsBundle(values={"ugbw_hz": MetricValue(name="ugbw_hz", value=2.0, unit="Hz")})

    result = evaluate_objectives(spec, metrics)

    assert result["per_objective"][0]["value"] == 2.0
    assert result["per_objective"][0]["passed"] is True
