from pathlib import Path

from eesizer_core.config import SimulationConfig
from eesizer_core.simulation import MockNgSpiceSimulator


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
