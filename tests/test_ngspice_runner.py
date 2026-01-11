import subprocess

import pytest

from eesizer_core.contracts.enums import SimKind
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim import NgspiceRunOperator, SpiceDeck
from eesizer_core.contracts.errors import SimulationError


def test_ngspice_runner_success(monkeypatch, tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(text="* test\n.end\n", kind=SimKind.ac, expected_outputs={"ac_csv": "ac.csv"})

    def fake_run(cmd, capture_output, text, check, timeout, cwd):
        # Simulate ngspice writing expected outputs into the cwd.
        for rel_path in deck.expected_outputs.values():
            out_path = cwd / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("freq,mag\n1,0.0\n", encoding="utf-8")
        (cwd / "ngspice_ac.log").write_text("log", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    op = NgspiceRunOperator(ngspice_bin="ngspice")
    result = op.run({"deck": deck}, ctx)

    raw = result.outputs["raw_data"]
    assert raw.outputs["ac_csv"].exists()
    assert raw.log_path.exists()
    assert "deck_ac.sp" in str(result.outputs["deck_path"])
    assert raw.returncode == 0


def test_ngspice_runner_missing_binary(monkeypatch, tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(text="* test\n.end\n", kind=SimKind.dc, expected_outputs={})

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("ngspice")

    monkeypatch.setattr(subprocess, "run", fake_run)

    op = NgspiceRunOperator(ngspice_bin="/missing/ngspice")

    with pytest.raises(SimulationError, match="ngspice executable not found"):
        op.run({"deck": deck}, ctx)


def test_ngspice_runner_missing_expected_output(monkeypatch, tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(text="* test\n.end\n", kind=SimKind.tran, expected_outputs={"tran_csv": "tran.csv"})

    def fake_run(cmd, capture_output, text, check, timeout, cwd):
        # Do NOT create expected outputs to simulate a deck problem.
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    op = NgspiceRunOperator()

    with pytest.raises(SimulationError, match="Expected output"):
        op.run({"deck": deck}, ctx)
