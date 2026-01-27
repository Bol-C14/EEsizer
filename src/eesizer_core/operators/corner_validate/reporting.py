from __future__ import annotations

from typing import Any, Iterable, Mapping
import math


def replace_section(report_text: str, heading: str, new_section: list[str]) -> str:
    if not new_section:
        return report_text
    lines = report_text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        if lines and lines[-1].strip():
            lines.append("")
        return "\n".join(lines + new_section).rstrip() + "\n"

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    updated = lines[:start] + new_section + lines[end:]
    return "\n".join(updated).rstrip() + "\n"


def _fmt_float(value: Any, *, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    if abs(val) >= 1e3 or (abs(val) > 0 and abs(val) < 1e-2):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"


def _pct_drop(nominal: Any, worst: Any) -> str:
    try:
        n = float(nominal)
        w = float(worst)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(n) or not math.isfinite(w) or n == 0:
        return "n/a"
    return f"{(w - n) / n * 100.0:+.1f}%"


def build_robustness_section(
    *,
    config: Mapping[str, Any],
    corner_param_ids: Iterable[str],
    corner_count: int,
    robust_topk: list[Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = ["## Robustness Validation", ""]

    lines.append("Corner config:")
    lines.append(f"- candidates_source: {config.get('candidates_source', 'topk')}")
    lines.append(f"- corners: {config.get('corners', 'oat')}")
    lines.append(f"- override_mode: {config.get('override_mode', 'add')}")
    lines.append(f"- clamp_corner_overrides: {bool(config.get('clamp_corner_overrides', True))}")
    if config.get("corners") == "oat_topm":
        lines.append(f"- top_m: {int(config.get('top_m', 3))}")
    lines.append(f"- corner_param_ids: {', '.join(list(corner_param_ids)) if corner_param_ids else '(none)'}")
    lines.append(f"- corner_count: {corner_count}")
    lines.append("")

    if not robust_topk:
        lines.append("No candidates validated.")
        return lines

    lines.append("Robust Top-K (sorted by worst_score):")
    lines.append("")
    headers = [
        "iter",
        "pass_rate",
        "worst_score",
        "worst_corner",
        "nom_ugbw_hz",
        "nom_pm_deg",
        "nom_power_w",
        "worst_ugbw_hz",
        "worst_pm_deg",
        "worst_power_w",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for entry in robust_topk:
        nominal = entry.get("nominal_metrics") or {}
        worst = entry.get("worst_metrics") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.get("iteration", "")),
                    _fmt_float(entry.get("pass_rate"), digits=3),
                    _fmt_float(entry.get("worst_score"), digits=3),
                    str(entry.get("worst_corner_id") or ""),
                    _fmt_float(nominal.get("ugbw_hz")),
                    _fmt_float(nominal.get("phase_margin_deg")),
                    _fmt_float(nominal.get("power_w")),
                    _fmt_float(worst.get("ugbw_hz")),
                    _fmt_float(worst.get("phase_margin_deg")),
                    _fmt_float(worst.get("power_w")),
                ]
            )
            + " |"
        )

    # Paper-friendly comparison sentence when possible.
    if len(robust_topk) >= 1:
        top1 = robust_topk[0]
        nominal = top1.get("nominal_metrics") or {}
        worst = top1.get("worst_metrics") or {}
        lines.append("")
        lines.append(
            "Top-1 worst-case deltas: "
            f"UGBW {_pct_drop(nominal.get('ugbw_hz'), worst.get('ugbw_hz'))}, "
            f"Power {_pct_drop(nominal.get('power_w'), worst.get('power_w'))}."
        )

    return lines

