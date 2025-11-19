import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "pipeline.run"]


def test_cli_runs_end_to_end(tmp_path):
    netlist_source = REPO_ROOT / "initial_circuit_netlist" / "inv.cir"
    netlist_copy = tmp_path / "inv.cir"
    netlist_copy.write_text(netlist_source.read_text())

    workdir = tmp_path / "work"
    result = subprocess.run(
        CLI
        + [
            "--netlist",
            str(netlist_copy),
            "--goal",
            "Demonstrate automated sizing",
            "--target-gain",
            "40",
            "--target-power",
            "3",
            "--workdir",
            str(workdir),
            "--run-id",
            "pytest",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["run_id"] == "pytest"
    assert payload["metrics"]["gain_db"] >= 40
    result_file = workdir / "pytest" / "pipeline_result.json"
    assert result_file.exists()
