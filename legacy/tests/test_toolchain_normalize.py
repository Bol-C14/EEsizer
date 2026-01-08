from agent_test_gpt.toolchain import normalize_tool_chain, validate_tool_chain


def _names(chain):
    return [c.get("name") for c in chain["tool_calls"]]


def test_normalize_drops_run_ngspice_and_injects_ac():
    tool_chain = {"tool_calls": [
        {"name": "dc_simulation"},
        {"name": "run_ngspice"},
        {"name": "bandwidth"},
    ]}

    normalized = normalize_tool_chain(tool_chain)
    names = _names(normalized)

    assert "run_ngspice" not in names
    assert names[0] == "ac_simulation"  # injected for bandwidth
    assert "dc_simulation" in names
    validate_tool_chain(normalized)


def test_normalize_injects_sim_for_analysis_only():
    tool_chain = {"tool_calls": [
        {"name": "bandwidth"},
        {"name": "unity_bandwidth"},
    ]}

    normalized = normalize_tool_chain(tool_chain)
    names = _names(normalized)

    assert names[0] == "ac_simulation"
    validate_tool_chain(normalized)


def test_normalize_dedup_simulations():
    tool_chain = {"tool_calls": [
        {"name": "dc_simulation"},
        {"name": "dc_simulation"},
        {"name": "output_swing"},
    ]}

    normalized = normalize_tool_chain(tool_chain)
    names = _names(normalized)

    assert names.count("dc_simulation") == 1
    validate_tool_chain(normalized)
