from eesizer_core.contracts import CircuitSpec, Objective
from eesizer_core.operators.spec_diff import diff_specs


def test_spec_diff_detects_objective_target_change() -> None:
    a = CircuitSpec(objectives=(Objective(metric="phase_margin_deg", target=55.0, sense="ge"),))
    b = CircuitSpec(objectives=(Objective(metric="phase_margin_deg", target=60.0, sense="ge"),))
    diff = diff_specs(a, b)

    assert diff["objectives_changed"]
    row = diff["objectives_changed"][0]
    assert row["metric"] == "phase_margin_deg"
    assert row["from"]["target"] == 55.0
    assert row["to"]["target"] == 60.0

