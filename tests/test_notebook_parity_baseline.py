"""Regression test to mirror the notebook baseline behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eesizer_core.agents.simple import OptimizationTargets
from tests.helpers.pipeline_runner import (
    PipelineRunResult,
    has_real_ngspice,
    run_pipeline_for_test,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_NETLIST = REPO_ROOT / "initial_circuit_netlist" / "ota.cir"
GOLDEN_METRICS = json.loads(
    (REPO_ROOT / "tests" / "golden" / "notebook_parity_metrics.json").read_text()
)


@pytest.mark.parametrize("use_real", [has_real_ngspice()])
def test_notebook_parity_baseline(tmp_path: Path, use_real: bool) -> None:
    if use_real is False:
        pytest.skip("Real ngspice not available; baseline numeric check skipped")

    result = _run_baseline(tmp_path, use_real_ngspice=use_real)

    _assert_core_artifacts(result)
    _assert_metrics_close(result.summary["metrics"], GOLDEN_METRICS)


def test_notebook_parity_with_mock(tmp_path: Path) -> None:
    result = _run_baseline(tmp_path, use_real_ngspice=False)

    _assert_core_artifacts(result)
    metrics = result.summary["metrics"]
    assert set(GOLDEN_METRICS).issubset(metrics)


def _run_baseline(tmp_path: Path, *, use_real_ngspice: bool) -> PipelineRunResult:
    workdir = tmp_path / "baseline_run"
    targets = OptimizationTargets(gain_db=55.0, power_mw=5.0)
    return run_pipeline_for_test(
        netlist=BASELINE_NETLIST,
        goal="Validate OTA sizing pipeline",
        targets=targets,
        run_id="baseline", 
        workdir=workdir,
        use_real_ngspice=use_real_ngspice,
    )


def _assert_core_artifacts(result: PipelineRunResult) -> None:
    summary = result.summary
    result_path = result.run_dir / "pipeline_result.json"
    assert result_path.exists()
    assert summary["run_id"] == "baseline"
    artifacts = summary["artifacts"]
    for key in [
        "netlist_copy",
        "simulation_metrics",
        "simulation_summary",
        "optimization_summary",
        "optimization_history_csv",
        "optimization_history_log",
        "optimization_history_pdf",
    ]:
        assert key in artifacts
        assert Path(artifacts[key]["path"]).exists()


def _assert_metrics_close(observed: dict, expected: dict) -> None:
    for key, expected_value in expected.items():
        assert key in observed
        if isinstance(expected_value, (int, float)):
            assert observed[key] == pytest.approx(expected_value, rel=0.1, abs=1e-3)
        else:
            assert observed[key] == expected_value
