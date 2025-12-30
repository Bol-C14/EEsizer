import copy

from agent_test_gpt import optimization


def test_parse_target_values_no_globals_mutation():
    sample = """
    {
      "target_values": [
        {
          "ac_gain_target": "30dB",
          "bandwidth_target": "1e6Hz",
          "thd_target": "-30dB"
        }
      ]
    }
    """
    before = set(optimization.__dict__.keys())
    parsed = optimization.parse_target_values(sample, lambda x: float(str(x).replace("dB", "").replace("Hz", "")))
    after = set(optimization.__dict__.keys())

    assert before == after, "parse_target_values should not mutate module globals"
    assert parsed.targets["gain_target"] == 30.0
    assert parsed.targets["bandwidth_target"] == 1_000_000.0
    assert parsed.targets["thd_target"] == -30.0
    assert parsed.passes["gain_pass"] is False
    assert parsed.passes["bw_pass"] is False
