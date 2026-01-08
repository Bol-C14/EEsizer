import math
import numpy as np

from agent_test_gpt.simulation_utils import _calc_output_swing, _gain_db, _first_crossing, bandwidth, unity_bandwidth


def _write_ac_dat(tmp_path, freq, vec, name="output_ac.dat"):
    data = np.column_stack([freq, np.real(vec), np.imag(vec)])
    lines = ["freq re im"]
    for row in data:
        lines.append(f"{row[0]:.6g} {row[1]:.6g} {row[2]:.6g}")
    path = tmp_path / name
    path.write_text("\n".join(lines))
    return path


def test_bandwidth_and_ugbw_on_lowpass(tmp_path):
    fc = 1e3
    dc_gain = 100.0  # 40 dB
    freq = np.logspace(2, 6, 200)  # 1e2 to 1e6
    base_mag = 1.0 / np.sqrt(1.0 + (freq / fc) ** 2)
    vec = base_mag * dc_gain  # purely real response

    _write_ac_dat(tmp_path, freq, vec)

    bw = bandwidth("output_ac", output_dir=str(tmp_path))
    ugbw = unity_bandwidth("output_ac", output_dir=str(tmp_path))

    assert math.isclose(bw, fc, rel_tol=0.15)
    assert math.isclose(ugbw, fc * dc_gain, rel_tol=0.15)


def test_gain_helpers_crossing():
    freq = np.array([1.0, 2.0, 3.0])
    vec = np.array([10.0, 1.0, 0.1])
    gain_db = _gain_db(vec)
    # Crossing 0 dB between first two points
    crossing = _first_crossing(freq, gain_db, target_db=0, from_above=True)
    assert 1.0 <= crossing <= 2.0


def test_calc_output_swing_linear_region():
    gain_target = 10.0
    tol = 0.05
    in1 = np.linspace(-0.1, 0.1, 200)
    ideal = gain_target * in1
    saturated = np.clip(ideal, -0.6, 0.6)

    swing = _calc_output_swing(saturated, in1, gain_target, tol)
    assert math.isclose(swing, 1.2, rel_tol=0.05)
