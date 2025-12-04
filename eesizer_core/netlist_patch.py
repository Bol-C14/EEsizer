"""Lightweight netlist patcher to apply structured param changes without LLM rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping


@dataclass(slots=True)
class ParamChange:
    """Single parameter change for a specific component."""

    component: str
    parameter: str
    operation: str  # one of {"set", "scale", "add"}
    value: float


def apply_param_changes(netlist_text: str, changes: Iterable[ParamChange]) -> tuple[str, MutableMapping[str, float]]:
    """Apply a list of parameter changes to the netlist text.

    Supports simple numeric replacements of the form ``W=1u`` or ``L=90n`` on
    component lines. Unknown components/params are skipped.
    Returns the patched text and a mapping of params that were updated.
    """

    lines = netlist_text.splitlines()
    applied: MutableMapping[str, float] = {}
    change_map = {}
    for change in changes:
        key = change.component.strip()
        change_map.setdefault(key.lower(), []).append(change)

    def _update_value(old_value: float, change: ParamChange) -> float:
        if change.operation == "set":
            return change.value
        if change.operation == "scale":
            return old_value * change.value
        if change.operation == "add":
            return old_value + change.value
        return old_value

    param_pattern_cache: MutableMapping[tuple[str, str], re.Pattern[str]] = {}

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue
        tokens = stripped.split()
        component_id = tokens[0].lower()
        if component_id not in change_map:
            continue
        original_line = lines[idx]
        for change in change_map[component_id]:
            pattern_key = (component_id, change.parameter.lower())
            if pattern_key not in param_pattern_cache:
                param_pattern_cache[pattern_key] = re.compile(
                    rf"(?P<prefix>\b{re.escape(change.parameter)}\s*=\s*)(?P<value>[-+]?[\d\.Ee]+)(?P<suffix>[a-zA-Z]*)",
                    re.IGNORECASE,
                )
            pattern = param_pattern_cache[pattern_key]
            match = pattern.search(original_line)
            if not match:
                if change.parameter.lower() not in {"r", "value"}:
                    continue
                tokens = original_line.split()
                replaced = False
                for t_idx in range(len(tokens) - 1, 0, -1):
                    scalar_match = re.match(r"(?P<value>[-+]?[\d\.Ee]+)(?P<suffix>[a-zA-Z]*)", tokens[t_idx])
                    if not scalar_match:
                        continue
                    raw_val = scalar_match.group("value")
                    suffix = scalar_match.group("suffix")
                    try:
                        old_val = _parse_with_suffix(raw_val, suffix)
                    except (TypeError, ValueError):
                        continue
                    new_val = _update_value(old_val, change)
                    tokens[t_idx] = _format_with_suffix(new_val, suffix)
                    replaced = True
                    applied[f"{change.component}.{change.parameter}"] = new_val
                    break
                if replaced:
                    original_line = " ".join(tokens)
                continue
            try:
                old_val = _parse_with_suffix(match.group("value"), match.group("suffix"))
            except (TypeError, ValueError):
                continue
            new_val = _update_value(old_val, change)
            replacement = f"{match.group('prefix')}{_format_with_suffix(new_val, match.group('suffix'))}"
            original_line = pattern.sub(replacement, original_line, count=1)
            applied[f"{change.component}.{change.parameter}"] = new_val
        lines[idx] = original_line

    return "\n".join(lines) + ("\n" if netlist_text.endswith("\n") else ""), applied


def _parse_with_suffix(value: str, suffix: str | None) -> float:
    multipliers = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
    }
    suffix = (suffix or "").lower()
    if suffix == "meg":
        factor = multipliers["meg"]
    elif suffix in multipliers:
        factor = multipliers[suffix]
    else:
        factor = 1.0
    return float(value) * factor


def _format_with_suffix(value: float, suffix: str | None) -> str:
    suffix = (suffix or "").lower()
    multipliers = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
    }
    if suffix in multipliers and multipliers[suffix] != 0:
        scaled = value / multipliers[suffix]
        return f"{scaled:g}{suffix}"
    return f"{value:g}"


__all__ = ["ParamChange", "apply_param_changes"]
