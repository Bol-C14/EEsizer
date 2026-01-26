from __future__ import annotations

from pathlib import Path

import numpy as np

from eesizer_core.contracts.enums import SimKind
from eesizer_core.metrics import ComputeMetricsOperator, DEFAULT_REGISTRY
from eesizer_core.sim.artifacts import RawSimData


def _write_ac_fixture(path: Path, freq_hz: np.ndarray, vout: np.ndarray) -> None:
    header = (
        "* frequency real(v(vout)) imag(v(vout)) "
        "real(v(vinp)) imag(v(vinp)) real(v(vinn)) imag(v(vinn))"
    )
    lines = [header]
    for f, vo in zip(freq_hz, vout):
        lines.append(
            f"{f:g} {vo.real:g} {vo.imag:g} 1 0 0 0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_missing_unity_crossing(tmp_path: Path) -> None:
    a0 = 0.1
    fp = 100.0
    freq = np.logspace(1, 7, 200)
    resp = a0 / (1.0 + 1j * (freq / fp))

    ac_path = tmp_path / "ac.csv"
    _write_ac_fixture(ac_path, freq, resp)

    log_path = tmp_path / "ngspice.log"
    log_path.write_text("", encoding="utf-8")

    raw = RawSimData(
        kind=SimKind.ac,
        run_dir=tmp_path,
        outputs={"ac_csv": ac_path},
        outputs_meta={
            "ac_csv": (
                "frequency",
                "real(v(vout))",
                "imag(v(vout))",
                "real(v(vinp))",
                "imag(v(vinp))",
                "real(v(vinn))",
                "imag(v(vinn))",
            )
        },
        log_path=log_path,
        cmdline=[],
        returncode=0,
    )

    op = ComputeMetricsOperator(registry=DEFAULT_REGISTRY)
    result = op.run({"raw_data": raw, "metric_names": ["ugbw_hz", "phase_margin_deg"]}, ctx=None)
    metrics = result.outputs["metrics"]

    ugbw = metrics.values["ugbw_hz"]
    pm = metrics.values["phase_margin_deg"]

    assert ugbw.value is None
    assert pm.value is None
    assert ugbw.details["status"] == "missing"
    assert ugbw.details["reason"] == "no_unity_crossing"
