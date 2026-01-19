from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from ..contracts.errors import MetricError


def _all_numeric(tokens: list[str]) -> bool:
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return False
    return True


def _normalize_header(line: str, comment_prefix: str) -> list[str]:
    if comment_prefix and line.startswith(comment_prefix):
        line = line[len(comment_prefix) :]
    return [tok.strip() for tok in line.strip().split() if tok.strip()]


def load_wrdata_table(
    path: Path,
    expected_columns: Sequence[str] | None = None,
    comment_prefix: str = "*",
) -> Tuple[list[str], pd.DataFrame]:
    """Robustly load an ngspice wrdata table (whitespace-separated).

    Handling:
    - Strip empty lines.
    - If first non-empty line starts with comment_prefix, use it as header (sans prefix).
    - Else, if the first line is non-numeric, treat as header.
    - Else, assume no header and use expected_columns if provided (must fit column count) else auto c0..cN.
    - If expected_columns is provided, it overrides header names while preserving column order.
    - Comment-prefixed lines in the data section are skipped.
    Raises MetricError on missing file, empty content, or column count mismatch.
    """
    if not path.exists():
        raise MetricError(f"wrdata file not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise MetricError(f"wrdata file is empty: {path}")

    header_tokens: list[str] | None = None
    data_start = 0

    for idx, line in enumerate(lines):
        if comment_prefix and line.startswith(comment_prefix):
            stripped = _normalize_header(line, comment_prefix)
            if stripped and any(
                t.lower().startswith(("freq", "time")) or "(" in t or ")" in t for t in stripped
            ):
                header_tokens = stripped
                data_start = idx + 1
                break
            continue
        tokens = line.split()
        if not tokens:
            continue
        if not _all_numeric(tokens):
            header_tokens = tokens
            data_start = idx + 1
        else:
            data_start = idx
        break

    data_lines: list[str] = []
    for ln in lines[data_start:]:
        if comment_prefix and ln.startswith(comment_prefix):
            continue
        data_lines.append(ln)

    if not data_lines:
        raise MetricError(f"No data rows found in wrdata file: {path}")

    sample_cols = len(data_lines[0].split())
    if expected_columns is not None:
        if len(expected_columns) > sample_cols:
            raise MetricError(
                f"Data columns ({sample_cols}) fewer than expected ({len(expected_columns)}) in {path}"
            )
        header_tokens = list(expected_columns) + [f"c{i}" for i in range(len(expected_columns), sample_cols)]
    elif header_tokens is None:
        header_tokens = [f"c{i}" for i in range(sample_cols)]

    csv_buf = StringIO("\n".join(data_lines))
    df = pd.read_csv(csv_buf, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(header_tokens):
        raise MetricError(
            f"Column count mismatch: header {len(header_tokens)} vs data {df.shape[1]} in {path}"
        )

    df.columns = header_tokens
    return header_tokens, df
