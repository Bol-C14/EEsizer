from eesizer_core.contracts.guards import GuardCheck, GuardReport


def test_guard_report_aggregates_hard_and_soft_checks():
    hard_ok = GuardCheck(name="hard_ok", ok=True, severity="hard")
    soft_fail = GuardCheck(name="soft_fail", ok=False, severity="soft", reasons=("warn",))
    report = GuardReport(checks=(hard_ok, soft_fail))
    assert report.ok is True
    assert report.hard_fails == ()
    assert report.soft_fails and report.soft_fails[0].name == "soft_fail"

    hard_fail = GuardCheck(name="hard_fail", ok=False, severity="hard", reasons=("bad",))
    report2 = GuardReport(checks=(hard_fail, soft_fail))
    assert report2.ok is False
    assert report2.hard_fails and report2.hard_fails[0].name == "hard_fail"
