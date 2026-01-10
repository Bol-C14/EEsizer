from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re


_CONTROL_BLOCK_PATTERN = re.compile(r"\.control.*?\.endc", re.IGNORECASE | re.DOTALL)
_INCLUDE_PATTERN = re.compile(r"^\s*\.include\s+['\"]?(?P<path>[^'\"\\s]+)['\"]?", re.IGNORECASE)


@dataclass(frozen=True)
class SanitizeResult:
    sanitized_text: str
    includes: Tuple[str, ...]
    warnings: Tuple[str, ...]


def _strip_control_blocks(netlist_text: str) -> str:
    return _CONTROL_BLOCK_PATTERN.sub("", netlist_text)


def normalize_spice_lines(lines: List[str]) -> List[str]:
    """Merge continuation lines starting with '+' into previous line."""
    normalized: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("+"):
            continuation = stripped.lstrip("+").strip()
            if normalized:
                prev = normalized.pop()
                normalized.append(f"{prev.rstrip()} {continuation}".strip())
            elif continuation:
                normalized.append(continuation)
            continue
        normalized.append(line)
    return normalized


def has_control_block(text: str) -> bool:
    return bool(_CONTROL_BLOCK_PATTERN.search(text))


def sanitize_spice_netlist(netlist_text: str, max_lines: int = 5000) -> SanitizeResult:
    """Strip unsafe constructs and return sanitized text, includes, and warnings."""
    warnings: List[str] = []

    text_no_ctrl = _strip_control_blocks(netlist_text)
    lines = normalize_spice_lines(text_no_ctrl.splitlines())

    if len(lines) > max_lines:
        raise ValueError(f"netlist too large ({len(lines)} lines > {max_lines})")

    safe_lines: List[str] = []
    includes: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            safe_lines.append(line)
            continue

        match = _INCLUDE_PATTERN.match(line)
        if match:
            inc_path = match.group("path")
            path_obj = Path(inc_path)
            if path_obj.is_absolute() or ".." in path_obj.parts:
                warnings.append(f"dropped unsafe include: {inc_path}")
                continue
            includes.append(inc_path)
            safe_lines.append(line)
            continue

        safe_lines.append(line)

    sanitized_text = "\n".join(safe_lines)
    if netlist_text.endswith("\n") and not sanitized_text.endswith("\n"):
        sanitized_text += "\n"

    return SanitizeResult(
        sanitized_text=sanitized_text,
        includes=tuple(includes),
        warnings=tuple(warnings),
    )
