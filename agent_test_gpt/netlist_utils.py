"""Netlist and parsing helpers."""

import json
import re
from pathlib import Path


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
    try:
        # Step 1: Clean the input string (remove backticks, "json", and extra whitespace)
        nodes = node.strip().strip('`')  # Remove leading/trailing backticks and whitespace
        nodes = nodes.replace("json", "").strip()  # Remove the word "json" and any extra whitespace

        # Step 2: Parse the JSON string into a dictionary
        node_name = json.loads(nodes)  # Use json.loads() for parsing strings

        # Step 3: Initialize lists to store input nodes, output nodes, and source names
        input_nodes = []
        output_nodes = []
        source_names = []

        # Step 4: Iterate through the "nodes" list (support multiple schema variants)
        for node_item in node_name.get("nodes", []):
            if not isinstance(node_item, dict):
                continue

            # Legacy format: {"input_node": "in1"}, {"source_name": "Vid"}
            if "input_node" in node_item:
                input_nodes.append(node_item["input_node"])
                continue
            if "output_node" in node_item:
                output_nodes.append(node_item["output_node"])
                continue
            if "source_name" in node_item:
                source_names.append(node_item["source_name"])
                continue

            # Newer format: {"input_nodes": [..]}, {"source_names": [..]}
            if "input_nodes" in node_item and isinstance(node_item["input_nodes"], list):
                input_nodes.extend([str(x) for x in node_item["input_nodes"]])
                continue
            if "source_names" in node_item and isinstance(node_item["source_names"], list):
                source_names.extend([str(x) for x in node_item["source_names"]])
                continue

        # Step 5: Return the extracted lists
        return input_nodes, output_nodes, source_names

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return [], [], []
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
        return [], [], []


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
