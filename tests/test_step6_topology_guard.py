from eesizer_core.operators.guards import TopologyGuardOperator


def test_topology_guard_ok_when_signatures_match():
    op = TopologyGuardOperator()
    check = op.run({"signature_before": "abc123", "signature_after": "abc123"}, ctx=None).outputs["check"]
    assert check.ok is True
    assert check.reasons == ()


def test_topology_guard_fails_when_signatures_differ():
    op = TopologyGuardOperator()
    check = op.run({"signature_before": "abc123", "signature_after": "def456"}, ctx=None).outputs["check"]
    assert check.ok is False
    assert check.reasons
