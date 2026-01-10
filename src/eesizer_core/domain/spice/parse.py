from __future__ import annotations

from typing import Dict, List, Tuple

from ...contracts.artifacts import CircuitIR, Element, TokenLoc
from .tokenize import tokenize_spice_line
from .sanitize_rules import normalize_spice_lines


def _strip_inline_comment(line: str) -> str:
    if ";" in line:
        return line.split(";", 1)[0]
    return line


def _parse_param_token(token: str, line_idx: int, token_idx: int) -> TokenLoc | None:
    if "=" not in token:
        return None
    key, _, _ = token.partition("=")
    key = key.strip()
    if not key:
        return None
    eq_pos = token.find("=")
    return TokenLoc(
        line_idx=line_idx,
        token_idx=token_idx,
        key=key.lower(),
        raw_token=token,
        value_span=(eq_pos + 1, len(token)),
    )


def index_spice_netlist(netlist_text: str, includes: Tuple[str, ...] | None = None) -> CircuitIR:
    """Index a SPICE netlist into line-anchored elements.

    Expects sanitized text (control blocks removed, includes filtered) for consistency.
    """
    lines = normalize_spice_lines(netlist_text.splitlines())
    elements: Dict[str, Element] = {}
    param_locs: Dict[str, TokenLoc] = {}
    includes_list: List[str] = list(includes or [])
    warnings: List[str] = []

    for line_idx, line in enumerate(lines):
        line = _strip_inline_comment(line)
        stripped = line.strip()
        if not stripped:
            continue
        # Skip simple comment lines
        if stripped.startswith("*") or stripped.startswith("//"):
            continue

        # Includes (only recorded if not provided externally)
        if stripped.lower().startswith(".include") and includes is None:
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1:
                includes_list.append(parts[1].strip().strip("'\""))
            else:
                warnings.append(f"line {line_idx}: malformed include")
            continue

        # .param definitions
        if stripped.lower().startswith(".param"):
            tokens = stripped.split()
            for idx, tok in enumerate(tokens[1:], start=1):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    param_id = f"param.{loc.key.lower()}"
                    param_locs[param_id] = TokenLoc(
                        line_idx=loc.line_idx,
                        token_idx=loc.token_idx,
                        key=param_id,
                        raw_token=loc.raw_token,
                        value_span=loc.value_span,
                    )
            continue

        # Ignore other dot-directives for now
        if stripped.startswith("."):
            continue

        tokens, spans = tokenize_spice_line(stripped)
        if not tokens:
            continue

        name = tokens[0]
        name_l = name.lower()
        kind = name[0].upper()

        if kind == "M":
            if len(tokens) < 6:
                warnings.append(f"line {line_idx}: MOS line too short")
                continue
            nodes = tuple(tokens[1:5])
            model = tokens[5]
            param_tokens = tokens[6:]
            params: Dict[str, TokenLoc] = {}
            for idx, tok in enumerate(param_tokens, start=6):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    params[loc.key] = loc
                    param_locs[f"{name_l}.{loc.key}"] = loc
            elem = Element(name=name, etype="MOS", nodes=nodes, model_or_subckt=model, line_idx=line_idx, params=params)

        elif kind in {"R", "C", "L", "V", "I"}:
            if len(tokens) < 3:
                warnings.append(f"line {line_idx}: passive/source line too short")
                continue
            nodes = tuple(tokens[1:3])
            param_tokens = tokens[3:]
            params: Dict[str, TokenLoc] = {}

            if param_tokens:
                first = param_tokens[0]
                if "=" not in first:
                    loc = TokenLoc(
                        line_idx=line_idx,
                        token_idx=3,
                        key="value",
                        raw_token=first,
                        value_span=(0, len(first)),
                    )
                    params["value"] = loc
                    param_locs[f"{name_l}.value"] = loc
                    param_tokens = param_tokens[1:]

            for idx, tok in enumerate(param_tokens, start=len(tokens) - len(param_tokens)):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    params[loc.key] = loc
                    param_locs[f"{name_l}.{loc.key}"] = loc
            elem = Element(name=name, etype=kind, nodes=nodes, line_idx=line_idx, params=params)

        elif kind == "X":
            rest = tokens[1:]
            if not rest:
                warnings.append(f"line {line_idx}: subckt call missing content")
                continue
            param_start = next((i for i, t in enumerate(rest) if "=" in t), len(rest))
            non_param = rest[:param_start]
            param_tokens = rest[param_start:]
            if not non_param:
                warnings.append(f"line {line_idx}: subckt call missing subckt name")
                continue
            subckt = non_param[-1]
            nodes = tuple(non_param[:-1])
            params: Dict[str, TokenLoc] = {}
            for idx, tok in enumerate(param_tokens, start=len(tokens) - len(param_tokens)):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    params[loc.key] = loc
                    param_locs[f"{name_l}.{loc.key}"] = loc
            elem = Element(name=name, etype="SUBCKT", nodes=nodes, model_or_subckt=subckt, line_idx=line_idx, params=params)

        else:
            warnings.append(f"line {line_idx}: unsupported element '{name}'")
            continue

        elements[name] = elem

    return CircuitIR(
        lines=tuple(lines),
        elements=elements,
        param_locs=param_locs,
        includes=tuple(includes_list),
        warnings=tuple(warnings),
    )
