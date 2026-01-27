from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

from ..runtime.session_store import SessionStore


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _relpath(from_dir: Path, to_path: Path) -> str:
    try:
        rel = os.path.relpath(to_path, start=from_dir)
    except Exception:
        rel = str(to_path)
    return rel.replace("\\", "/")


def _fmt_hash(value: Any) -> str:
    if not value:
        return "-"
    text = str(value)
    return text if len(text) <= 20 else text[:8] + "…" + text[-8:]


def build_meta_report(store: SessionStore) -> str:
    state = store.load_session_state()
    session_dir = store.session_dir

    lines: list[str] = []
    lines.append("# Session Meta Report")
    lines.append("")
    lines.append("## Session")
    lines.append(f"- session_id: {state.session_id}")
    lines.append(f"- bench_id: {state.bench_id}")
    lines.append(f"- seed: {state.seed}")
    lines.append(f"- created_at: {state.created_at}")
    lines.append("")

    spec_trace = _read_jsonl(store.spec_trace_path)
    cfg_trace = _read_jsonl(store.cfg_trace_path)

    lines.append("## Spec Trace")
    if not spec_trace:
        lines.append("- (none)")
    else:
        lines.append("| rev | timestamp | reason | hash | linked_phase |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in sorted(spec_trace, key=lambda e: int(e.get("rev") or 0)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(entry.get("rev", "")),
                        str(entry.get("timestamp", "")),
                        str(entry.get("reason", "")),
                        _fmt_hash(entry.get("new_hash")),
                        str(entry.get("linked_phase") or ""),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.append("## Cfg Trace")
    if not cfg_trace:
        lines.append("- (none)")
    else:
        lines.append("| rev | timestamp | reason | hash | linked_phase |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in sorted(cfg_trace, key=lambda e: int(e.get("rev") or 0)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(entry.get("rev", "")),
                        str(entry.get("timestamp", "")),
                        str(entry.get("reason", "")),
                        _fmt_hash(entry.get("new_hash")),
                        str(entry.get("linked_phase") or ""),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.append("## Phases")
    if not state.phases:
        lines.append("- (no phases recorded yet)")
    else:
        lines.append("| phase | status | run | stop_reason | summary |")
        lines.append("| --- | --- | --- | --- | --- |")
        for phase in sorted(state.phases, key=lambda p: p.phase_id):
            run_dir = phase.run_dir
            if run_dir:
                report_path = Path(run_dir) / "report.md"
                run_link = _relpath(session_dir, report_path)
                run_cell = f"[report]({run_link})"
            else:
                run_cell = "-"
            stop_reason = phase.stop_reason or "-"
            summary = phase.output_summary or {}
            best_score = summary.get("best_score")
            summary_cell = f"best_score={best_score}" if best_score is not None else "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        phase.phase_id,
                        phase.status,
                        run_cell,
                        str(stop_reason),
                        summary_cell,
                    ]
                )
                + " |"
            )
    lines.append("")

    narrative_path = session_dir / "llm" / "narrative.md"
    if narrative_path.exists():
        lines.append("## LLM Interpretation (Optional)")
        lines.append(f"- narrative: {_relpath(session_dir, narrative_path)}")
        if state.latest_advice_rev is not None:
            lines.append(f"- latest_advice_rev: {state.latest_advice_rev}")
        lines.append("")

    plan_dir = session_dir / "llm" / "plan_advice"
    if state.latest_plan_rev is not None:
        latest_plan = plan_dir / f"plan_rev{int(state.latest_plan_rev):04d}"
        plan_opts = latest_plan / "plan_options.json"
        status = latest_plan / "status.json"
        if plan_opts.exists():
            lines.append("## LLM Plan (Optional)")
            lines.append(f"- latest_plan_rev: {state.latest_plan_rev}")
            lines.append(f"- plan_options: {_relpath(session_dir, plan_opts)}")
            if status.exists():
                lines.append(f"- plan_status: {_relpath(session_dir, status)}")
            lines.append("")

    # Robustness summary (if present in latest grid run).
    grid_ck = store.load_checkpoint("p1_grid") or {}
    grid_run_dir = grid_ck.get("run_dir")
    if grid_run_dir:
        robust = _read_json(Path(grid_run_dir) / "search" / "robust_topk.json")
        if isinstance(robust, list) and robust:
            top1 = robust[0]
            if isinstance(top1, dict):
                lines.append("## Robustness (Top-1)")
                lines.append(f"- iteration: {top1.get('iteration')}")
                lines.append(f"- worst_corner_id: {top1.get('worst_corner_id')}")
                nom = top1.get("nominal_metrics") or {}
                worst = top1.get("worst_metrics") or {}
                if isinstance(nom, dict) and isinstance(worst, dict):
                    lines.append(f"- nominal: ugbw_hz={nom.get('ugbw_hz')}, pm={nom.get('phase_margin_deg')}, power={nom.get('power_w')}")
                    lines.append(f"- worst: ugbw_hz={worst.get('ugbw_hz')}, pm={worst.get('phase_margin_deg')}, power={worst.get('power_w')}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"
