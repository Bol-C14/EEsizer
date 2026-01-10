from __future__ import annotations

from typing import List, Sequence, Tuple
import re


def tokenize_spice_line(line: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize a SPICE line, returning tokens and their (start, end) spans."""
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"\S+", line):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def tokens_at_indices(tokens: Sequence[str], spans: Sequence[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int]]]:
    """Utility to pair tokens with spans."""
    return list(zip(tokens, spans))
