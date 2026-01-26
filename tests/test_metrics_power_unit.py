from __future__ import annotations

from pathlib import Path

import pytest

from eesizer_core.contracts.enums import SimKind
from eesizer_core.metrics import ComputeMetricsOperator, DEFAULT_REGISTRY
from eesizer_core.sim.artifacts import RawSimData


def test_power_metric(tmp_path: Path) -> None:
    dc_path = tmp_path / "dc.csv"
    dc_path.write_text(
        "* v(vdd) v(vout) i(VDD)\n"
        "1.8 0.9 -0.001\n",
        encoding="utf-8",
    )

    log_path = tmp_path / "ngspice.log"
    log_path.write_text("", encoding="utf-8")

    raw = RawSimData(
        kind=SimKind.dc,
        run_dir=tmp_path,
        outputs={"dc_csv": dc_path},
        outputs_meta={"dc_csv": ("v(vdd)", "v(vout)", "i(VDD)")},
        log_path=log_path,
        cmdline=[],
        returncode=0,
    )

    op = ComputeMetricsOperator(registry=DEFAULT_REGISTRY)
    result = op.run({"raw_data": raw, "metric_names": ["power_w"]}, ctx=None)
    metrics = result.outputs["metrics"]

    power = metrics.values["power_w"].value
    assert power == pytest.approx(1.8e-3)
