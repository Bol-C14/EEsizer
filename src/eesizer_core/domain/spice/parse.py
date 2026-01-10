from __future__ import annotations

from typing import Dict, List, Tuple

from ...contracts.artifacts import CircuitIR, Element, TokenLoc


def _merge_continuations(lines: List[str]) -> List[str]:
    merged: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("+"):
            continuation = stripped.lstrip("+").strip()
            if merged:
                prev = merged.pop()
                merged.append(f"{prev.rstrip()} {continuation}".strip())
            elif continuation:
                merged.append(continuation)
            continue
        merged.append(line)
    return merged


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
    lines = _merge_continuations(netlist_text.splitlines())
    elements: Dict[str, Element] = {}
    param_locs: Dict[str, TokenLoc] = {}
    includes_list: List[str] = list(includes or [])
    warnings: List[str] = []

    for line_idx, line in enumerate(lines):
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

        # Ignore other dot-directives for now
        if stripped.startswith("."):
            continue

        tokens = stripped.split()
        if not tokens:
            continue

        name = tokens[0]
        kind = name[0].upper()

        if kind == "M":
            if len(tokens) < 6:
                warnings.append(f"line {line_idx}: MOS line too short")
                continue
            nodes = tuple(tokens[1:5])
            model = tokens[5]
            param_tokens = tokens[6:]
            elem = Element(name=name, etype="MOS", nodes=nodes, model_or_subckt=model, line_idx=line_idx)
            for idx, tok in enumerate(param_tokens, start=6):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    elem.params[loc.key] = loc
                    param_locs[f"{name}.{loc.key}"] = loc

        elif kind in {"R", "C", "L", "V", "I"}:
            if len(tokens) < 3:
                warnings.append(f"line {line_idx}: passive/source line too short")
                continue
            nodes = tuple(tokens[1:3])
            param_tokens = tokens[3:]
            elem = Element(name=name, etype=kind, nodes=nodes, line_idx=line_idx)
            for idx, tok in enumerate(param_tokens, start=3):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    elem.params[loc.key] = loc
                    param_locs[f"{name}.{loc.key}"] = loc

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
            elem = Element(name=name, etype="SUBCKT", nodes=nodes, model_or_subckt=subckt, line_idx=line_idx)
            for idx, tok in enumerate(param_tokens, start=len(tokens) - len(param_tokens)):
                loc = _parse_param_token(tok, line_idx=line_idx, token_idx=idx)
                if loc:
                    elem.params[loc.key] = loc
                    param_locs[f"{name}.{loc.key}"] = loc

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
