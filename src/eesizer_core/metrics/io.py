from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..contracts.errors import MetricError


def _all_numeric(tokens: list[str]) -> bool:
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return False
    return True


def _normalize_header_line(line: str, comment_prefix: str = "*") -> list[str]:
    if line.startswith(comment_prefix):
        line = line[len(comment_prefix):]
    return [tok.strip() for tok in line.strip().split() if tok.strip()]


def load_wrdata_table(
    path: Path,
    expected_columns: Sequence[str] | None = None,
    comment_prefix: str = "*",
) -> pd.DataFrame:
    """Robustly load an ngspice wrdata table that may or may not include a header.

    Rules:
    - Strip empty lines.
    - If the first non-empty line starts with comment_prefix, treat it as the header (sans prefix).
    - Else, if the first line is non-numeric, treat it as the header.
    - Else, assume no header; use expected_columns if shape matches, otherwise auto-generate c0..cN.
    - Comment-prefixed lines in the data section are skipped.
    """
    if not path.exists():
        raise MetricError(f"wrdata file not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise MetricError(f"wrdata file is empty: {path}")

    header_tokens: list[str] | None = None
    data_start = 0

    first = lines[0]
    if first.startswith(comment_prefix):
        header_tokens = _normalize_header_line(first, comment_prefix=comment_prefix)
        data_start = 1
    else:
        tokens = first.split()
        if not _all_numeric(tokens):
            header_tokens = tokens
            data_start = 1

    data_lines: list[str] = []
    for ln in lines[data_start:]:
        if comment_prefix and ln.startswith(comment_prefix):
            continue
        data_lines.append(ln)

    if not data_lines:
        raise MetricError(f"No data rows found in wrdata file: {path}")

    sample_cols = len(data_lines[0].split())
    if header_tokens is None:
        if expected_columns and len(expected_columns) == sample_cols:
            header_tokens = list(expected_columns)
        else:
            header_tokens = [f"c{i}" for i in range(sample_cols)]
    else:
        if expected_columns and len(expected_columns) != len(header_tokens):
            raise MetricError(
                f"Header column count ({len(header_tokens)}) does not match expected ({len(expected_columns)})"
            )

    csv_buf = StringIO("\n".join(data_lines))
    df = pd.read_csv(csv_buf, sep=r"\s+", header=None, engine="python")

    if df.shape[1] != len(header_tokens):
        raise MetricError(
            f"Column count mismatch: header {len(header_tokens)} vs data {df.shape[1]} in {path}"
        )

    df.columns = header_tokens
    return df
