"""Netlist and parsing helpers."""

import json
import re
from pathlib import Path

from agent_test_gpt import config
from agent_test_gpt.logging_utils import get_logger


_logger = get_logger(__name__)


_INCLUDE_PATTERN = re.compile(r"^\s*\.include\s+['\"]?(?P<path>[^'\"\s]+)['\"]?", re.IGNORECASE | re.MULTILINE)
_CONTROL_BLOCK_PATTERN = re.compile(r"\.control.*?\.endc", re.IGNORECASE | re.DOTALL)


def _resolve_spice_include_path(raw_path: str) -> str | None:
    """Resolve a relative .include path against known project folders.

    This keeps notebook-derived netlists (e.g. `.include 'ptm_90.txt'`) runnable
    when executed from the repository root.
    """

    include_path = Path(raw_path)
    if include_path.is_absolute():
        return raw_path if include_path.exists() else None

    cwd = Path.cwd()
    if (cwd / include_path).exists():
        return raw_path

    search_roots = (
        config.RESOURCE_DIR,
        cwd / "agent_test_gpt",
        cwd / "agent_test_gemini",
        cwd / "agent_test_claude",
        cwd / "legacy_notebook" / "agent_test_gpt",
        cwd / "legacy_notebook" / "agent_test_gemini",
        cwd / "legacy_notebook" / "agent_test_claude",
        cwd / "variation",
    )
    for root in search_roots:
        candidate = root / include_path
        if candidate.exists():
            try:
                return str(candidate.relative_to(cwd))
            except ValueError:
                return str(candidate)
    return None


def normalize_spice_includes(netlist_text: str) -> str:
    """Rewrite `.include` lines to point at resolvable files when possible."""

    def _repl(match: re.Match[str]) -> str:
        raw_path = match.group("path")
        resolved = _resolve_spice_include_path(raw_path)
        if not resolved or resolved == raw_path:
            return match.group(0)
        return match.group(0).replace(raw_path, resolved, 1)

    return _INCLUDE_PATTERN.sub(_repl, netlist_text)


def strip_control_blocks(netlist_text: str) -> str:
    """Remove any existing .control/.endc blocks to avoid LLM-injected commands."""
    return _CONTROL_BLOCK_PATTERN.sub("", netlist_text)


def _is_safe_include(path: str) -> bool:
    # Reject absolute paths or traversal attempts
    p = Path(path)
    if p.is_absolute():
        return False
    parts = p.parts
    return ".." not in parts


def sanitize_netlist(netlist_text: str, max_lines: int = 20000) -> str:
    """Strip unsafe constructs from an LLM/user-provided netlist before simulation.

    - Remove any .control/.endc blocks (we inject our own).
    - Drop .include lines that point to absolute paths or contain traversal.
    - Enforce a line-count ceiling to avoid runaway payloads.
    """
    netlist_text = strip_control_blocks(netlist_text)

    safe_lines: list[str] = []
    for line in netlist_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(".include"):
            match = _INCLUDE_PATTERN.match(line)
            if not match:
                # Skip malformed include
                continue
            inc_path = match.group("path")
            if not _is_safe_include(inc_path):
                # Drop unsafe include
                continue
        safe_lines.append(line)

    if len(safe_lines) > max_lines:
        raise ValueError(f"netlist too large ({len(safe_lines)} lines > {max_lines})")

    return "\n".join(safe_lines) + ("\n" if not netlist_text.endswith("\n") else "")


def nodes_extract(node):
    """Parse node JSON into flat lists; raise if required fields are missing."""

    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)] if str(value).strip() else []

    try:
        cleaned = node.strip().strip('`').replace("json", "").strip()
        payload = json.loads(cleaned)
    except Exception as exc:  # includes JSONDecodeError and others
        _logger.error("Failed to parse node JSON: %s", exc)
        raise

    input_nodes: list[str] = []
    output_nodes: list[str] = []
    source_names: list[str] = []

    # Allow top-level dict fields as well as a "nodes" list of dicts
    candidate_items = []
    if isinstance(payload, dict):
        candidate_items.extend(payload.get("nodes", []))
        candidate_items.append(payload)
    elif isinstance(payload, list):
        candidate_items.extend(payload)
    else:
        _logger.error("Unexpected nodes payload type: %s", type(payload).__name__)
        raise ValueError(f"Unexpected nodes payload type: {type(payload).__name__}")

    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        # normalize keys to lower for matching
        lower_keys = {k.lower(): v for k, v in item.items()}

        input_nodes.extend(_as_list(lower_keys.get("input_node")))
        input_nodes.extend(_as_list(lower_keys.get("input_nodes")))

        output_nodes.extend(_as_list(lower_keys.get("output_node")))
        output_nodes.extend(_as_list(lower_keys.get("output_nodes")))

        # accept various source key names
        source_names.extend(_as_list(lower_keys.get("source_name")))
        source_names.extend(_as_list(lower_keys.get("source_names")))
        source_names.extend(_as_list(lower_keys.get("ac_source_name")))
        source_names.extend(_as_list(lower_keys.get("ac_source_names")))
        source_names.extend(_as_list(lower_keys.get("tran_source_name")))
        source_names.extend(_as_list(lower_keys.get("tran_source_names")))
        source_names.extend(_as_list(lower_keys.get("supply_source_name")))
        source_names.extend(_as_list(lower_keys.get("supply_source_names")))

    # Deduplicate while preserving order
    def _dedup(seq: list[str]) -> list[str]:
        seen = set()
        result = []
        for val in seq:
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    input_nodes = _dedup(input_nodes)
    output_nodes = _dedup(output_nodes)
    source_names = _dedup(source_names)

    if not output_nodes:
        _logger.error("Node extraction produced no output nodes. Raw: %s", node)
        raise ValueError("output_nodes is empty after parsing node response")
    if not source_names:
        _logger.error("Node extraction produced no source names. Raw: %s", node)
        raise ValueError("source_names is empty after parsing node response")

    return input_nodes, output_nodes, source_names


def extract_code(text):
    regex1 = r"'''(.+?)'''"
    regex2 = r"```(.+?)```"

    matches1 = re.findall(regex1, text, re.DOTALL)
    matches2 = re.findall(regex2, text, re.DOTALL)

    extracted_code = "\n".join(matches1 + matches2)
    lines = extracted_code.split('\n')
    cleaned_lines = []

    for line in lines:
        if '*' in line:
            line = line.split('*')[0].strip()
        elif '#' in line:
            line = line.split('#')[0].strip()
        elif ';' in line:
            line = line.split(';')[0].strip()
        elif line.startswith('verilog'):
            line = '\n'
        if line:
            cleaned_lines.append(line)

    cleaned_code = "\n".join(cleaned_lines)
    return cleaned_code


def extract_number(value):
    # Convert the value to a string (in case it's an integer or None)
    value_str = str(value) if value is not None else "0"

    # Use regex to find all numeric values (including decimals)
    match = re.search(r"[-+]?\d*\.\d+|\d+", value_str)
    if match:
        return float(match.group(0))
    return None
