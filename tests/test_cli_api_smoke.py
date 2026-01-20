import sys

from eesizer_core import api, cli

def test_cli_main_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eesizer"])
    cli.main()
    captured = capsys.readouterr()
    assert "EEsizer CLI" in (captured.out + captured.err)


def test_api_imports_and_constructs():
    src = api.CircuitSource(kind=api.SourceKind.spice_netlist, text="R1 in out 1k\n")
    assert src.kind == api.SourceKind.spice_netlist
