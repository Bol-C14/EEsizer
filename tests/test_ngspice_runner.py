import subprocess

import pytest

from eesizer_core.contracts.enums import SimKind
from eesizer_core.contracts.errors import ValidationError
from eesizer_core.runtime.context import RunContext
from eesizer_core.sim import NgspiceRunOperator, SpiceDeck
from eesizer_core.contracts.errors import SimulationError


def test_ngspice_runner_success(monkeypatch, tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(
        text="* test\n.end\n",
        kind=SimKind.ac,
        expected_outputs={"ac_csv": "ac.csv"},
        expected_outputs_meta={"ac_csv": ("frequency", "real(v(out))", "imag(v(out))")},
    )

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
    assert raw.outputs_meta["ac_csv"] == ("frequency", "real(v(out))", "imag(v(out))")


def test_ngspice_runner_rewrites_only_wrdata(tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck_text = """* ac.csv mention should stay
.control
wrdata __OUT__/ac.csv frequency real(v(out)) imag(v(out))
.endc
.end
"""
    deck = SpiceDeck(
        text=deck_text,
        kind=SimKind.ac,
        expected_outputs={"ac_csv": "ac.csv"},
        expected_outputs_meta={"ac_csv": ("frequency", "real(v(out))", "imag(v(out))")},
    )
    op = NgspiceRunOperator(ngspice_bin="ngspice")
    stage_dir = ctx.run_dir() / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    deck_path = op._write_deck(deck, stage_dir, {"ac_csv": stage_dir / "ac.csv"})
    content = deck_path.read_text(encoding="utf-8")
    assert "__OUT__" not in content
    assert str((ctx.run_dir() / "stage" / "ac.csv")) in content
    # comment text unchanged
    assert "* ac.csv mention should stay" in content


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


def test_ngspice_runner_nonzero_returncode(monkeypatch, tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(text="* test\n.end\n", kind=SimKind.ac, expected_outputs={"ac_csv": "ac.csv"})

    def fake_run(cmd, capture_output, text, check, timeout, cwd):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="fail", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    op = NgspiceRunOperator()
    with pytest.raises(SimulationError, match="exited with code 1"):
        op.run({"deck": deck}, ctx)


def test_ngspice_runner_rejects_traversal_outputs(tmp_path):
    workspace = tmp_path / "output"
    ctx = RunContext(workspace_root=workspace)
    deck = SpiceDeck(text="* test\n.end\n", kind=SimKind.ac, expected_outputs={"ac_csv": "../ac.csv"})

    op = NgspiceRunOperator()
    with pytest.raises(ValidationError):
        op.run({"deck": deck}, ctx)
