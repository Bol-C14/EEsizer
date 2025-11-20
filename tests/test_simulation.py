from pathlib import Path

from eesizer_core.config import SimulationConfig
from eesizer_core.simulation import MockNgSpiceSimulator, NgSpiceRunner


def test_mock_simulator_computes_metrics():
    netlist = """
* inverter
M1 out in 0 0 nch W=1u L=90n
M2 out in vdd vdd pch W=2u L=90n
R1 vdd out 1k
C1 out 0 1p
"""
    simulator = MockNgSpiceSimulator(SimulationConfig(binary_path=Path("ngspice")))
    metrics = simulator.run(netlist)

    assert metrics["gain_db"] > 25
    assert metrics["power_mw"] >= 0.4
    assert metrics["bandwidth_hz"] < 1_000_000
    assert metrics["transistor_count"] == 2


def test_ngspice_runner_waveform_and_op_parsing(tmp_path):
    runner = NgSpiceRunner(SimulationConfig(binary_path=Path("ngspice")))
    tran_path = tmp_path / "output_tran.dat"
    tran_path.write_text(
        """time v(out) vgs vth
0.0 0.1 0.35 0.3
1e-6 1.2 0.32 0.28
2e-6 -0.2 0.31 0.26
"""
    )
    vgscheck_path = tmp_path / "vgscheck.txt"
    vgscheck_path.write_text("m1 0.31 0.28\nm2 0.25 0.30\n")

    wave_metrics, artifacts = runner._collect_waveform_metrics(tmp_path)

    assert wave_metrics["output_swing_max"] == 1.2
    assert wave_metrics["output_swing_min"] == -0.2
    assert wave_metrics["output_swing_pp"] == 1.4
    assert wave_metrics["vgs_vth_margin_min"] < 0.03
    assert wave_metrics["vgs_vth_violations"] >= 1

    artifact_dir = tmp_path / "sim_artifacts"
    runner._persist_artifacts(artifacts, artifact_dir)
    assert (artifact_dir / "output_tran.dat").exists()
    assert (artifact_dir / "vgscheck.txt").exists()
