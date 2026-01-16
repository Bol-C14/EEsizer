import math

from eesizer_core.contracts.artifacts import CircuitSpec, MetricsBundle, MetricValue, Objective
from eesizer_core.operators.guards import BehaviorGuardOperator


def _make_spec():
    return CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))


def test_behavior_guard_rejects_missing_metric():
    spec = _make_spec()
    metrics = MetricsBundle()
    op = BehaviorGuardOperator()
    check = op.run({"metrics": metrics, "spec": spec, "stage_map": {}}, ctx=None).outputs["check"]
    assert not check.ok
    assert check.severity == "hard"
    assert any("missing" in reason for reason in check.reasons)


def test_behavior_guard_rejects_none_metric_value():
    spec = _make_spec()
    metrics = MetricsBundle(values={"ac_mag_db_at_1k": MetricValue(name="ac_mag_db_at_1k", value=None)})
    op = BehaviorGuardOperator()
    check = op.run({"metrics": metrics, "spec": spec, "stage_map": {}}, ctx=None).outputs["check"]
    assert not check.ok
    assert check.severity == "hard"
    assert any("None" in reason for reason in check.reasons)


def test_behavior_guard_rejects_nan_metric_value():
    spec = _make_spec()
    metrics = MetricsBundle(values={"ac_mag_db_at_1k": MetricValue(name="ac_mag_db_at_1k", value=math.nan)})
    op = BehaviorGuardOperator()
    check = op.run({"metrics": metrics, "spec": spec, "stage_map": {}}, ctx=None).outputs["check"]
    assert not check.ok
    assert check.severity == "hard"
    assert any("non-finite" in reason for reason in check.reasons)


def test_behavior_guard_scans_logs_for_fatal_patterns(tmp_path):
    spec = _make_spec()
    metrics = MetricsBundle(values={"ac_mag_db_at_1k": MetricValue(name="ac_mag_db_at_1k", value=-10.0)})
    log_path = tmp_path / "ngspice_ac.log"
    log_path.write_text("warning: singular matrix detected\n", encoding="utf-8")
    op = BehaviorGuardOperator()
    check = op.run({"metrics": metrics, "spec": spec, "stage_map": {"ac": str(tmp_path)}}, ctx=None).outputs["check"]
    assert not check.ok
    assert check.severity == "hard"
    assert any("singular matrix" in reason for reason in check.reasons)
