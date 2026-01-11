from pathlib import Path

from eesizer_core.io.ngspice_wrdata import load_wrdata_table


def test_wrdata_loader_handles_comments_and_header(tmp_path):
    path = tmp_path / "ac.dat"
    path.write_text(
        "\n*\n* comment line\n* frequency real(v(out)) imag(v(out))\n 1 1 0\n 10 0.1 0\n",
        encoding="utf-8",
    )
    cols, df = load_wrdata_table(path)
    assert cols == ["frequency", "real(v(out))", "imag(v(out))"]
    assert list(df.columns) == cols


def test_wrdata_loader_headerless(tmp_path):
    path = tmp_path / "dc.dat"
    path.write_text("0 0\n1 1\n", encoding="utf-8")
    expected = ["v(in)", "v(out)"]
    cols, df = load_wrdata_table(path, expected_columns=expected)
    assert cols[:2] == expected
    assert list(df.columns)[:2] == expected
