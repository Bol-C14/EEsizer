import pytest

from eesizer_core.contracts.errors import ValidationError
from eesizer_core.operators.netlist import SpiceSanitizeOperator


def test_sanitize_drops_absolute_include():
    netlist = ".include /abs/evil.sp\nR1 in out 1k\n"
    op = SpiceSanitizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)

    assert "/abs/evil.sp" not in result.outputs["sanitized_text"]
    assert "dropped unsafe include: /abs/evil.sp" in result.warnings
    assert result.outputs["includes"] == ()


def test_sanitize_drops_parent_include():
    netlist = ".include ../unsafe.sp\nR1 in out 1k\n"
    op = SpiceSanitizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)

    assert "../unsafe.sp" not in result.outputs["sanitized_text"]
    assert "dropped unsafe include: ../unsafe.sp" in result.warnings
    assert result.outputs["includes"] == ()


def test_sanitize_keeps_quoted_include_with_space():
    netlist = '.include "dir with space/ok.sp"\nR1 in out 1k\n'
    op = SpiceSanitizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)

    assert "dir with space/ok.sp" in result.outputs["includes"]
    assert 'include "dir with space/ok.sp"' in result.outputs["sanitized_text"]


def test_sanitize_drops_env_var_include():
    netlist = ".include $HOME/secret.sp\nR1 in out 1k\n"
    op = SpiceSanitizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)

    assert "$HOME/secret.sp" not in result.outputs["sanitized_text"]
    assert "dropped unsafe include: $HOME/secret.sp" in result.warnings
    assert result.outputs["includes"] == ()


def test_sanitize_removes_multiple_control_blocks():
    netlist = "\n".join(
        [
            ".control",
            "echo first",
            ".endc",
            "R1 in out 1k",
            ".CONTROL",
            "echo second",
            ".ENDC",
        ]
    )
    op = SpiceSanitizeOperator()
    result = op.run({"netlist_text": netlist}, ctx=None)

    assert ".control" not in result.outputs["sanitized_text"].lower()
    assert "R1 in out 1k" in result.outputs["sanitized_text"]


def test_sanitize_rejects_oversized_netlist():
    netlist = "R1 in out 1k\nR2 in out 2k\n"
    op = SpiceSanitizeOperator()

    with pytest.raises(ValidationError, match="netlist too large"):
        op.run({"netlist_text": netlist, "max_lines": 1}, ctx=None)
