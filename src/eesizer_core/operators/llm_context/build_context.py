from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

from ...runtime.session_store import SessionStore


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _clip_lines(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n..."


def _extract_section(report_text: str, heading: str, *, max_lines: int) -> str:
    lines = report_text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return _clip_lines("\n".join(lines[start:end]).strip(), max_lines=max_lines)


def build_session_llm_context(
    store: SessionStore,
    *,
    max_topk: int = 5,
    max_pareto: int = 5,
    max_report_section_lines: int = 60,
) -> dict[str, Any]:
    """Build a compact, deterministic LLM context from session + phase artifacts."""
    state = store.load_session_state()
    session_dir = store.session_dir

    evidence: list[str] = []

    meta_report_path = session_dir / "meta_report.md"
    meta_report = _read_text(meta_report_path)
    if meta_report:
        evidence.append("session/meta_report.md")

    ck_grid = store.load_checkpoint("p1_grid") or {}
    grid_run_dir = ck_grid.get("run_dir")
    grid_report = ""
    topk = []
    pareto = []
    robust_topk = []
    robust_pareto = []
    sensitivity = None

    report_sections: dict[str, str] = {}
    if grid_run_dir:
        run_dir = Path(grid_run_dir)
        report_path = run_dir / "report.md"
        grid_report = _read_text(report_path)
        if grid_report:
            evidence.append(f"{store.recorder.relpath(report_path)}")
            for heading in (
                "## Run Summary",
                "## Parameters Selected",
                "## Ranges & Discretization",
                "## Candidate Generation",
                "## Top-K Candidates",
                "## Pareto Front",
                "## Failures",
                "## Insights",
                "## Robustness Validation",
                "## Plots",
            ):
                excerpt = _extract_section(grid_report, heading, max_lines=max_report_section_lines)
                if excerpt:
                    report_sections[heading] = excerpt

        topk_payload = _read_json(run_dir / "search" / "topk.json")
        if isinstance(topk_payload, list):
            topk = topk_payload[:max_topk]
            evidence.append(store.recorder.relpath(run_dir / "search" / "topk.json"))

        pareto_payload = _read_json(run_dir / "search" / "pareto.json")
        if isinstance(pareto_payload, list):
            pareto = pareto_payload[:max_pareto]
            evidence.append(store.recorder.relpath(run_dir / "search" / "pareto.json"))

        robust_topk_payload = _read_json(run_dir / "search" / "robust_topk.json")
        if isinstance(robust_topk_payload, list):
            robust_topk = robust_topk_payload[:max_topk]
            evidence.append(store.recorder.relpath(run_dir / "search" / "robust_topk.json"))

        robust_pareto_payload = _read_json(run_dir / "search" / "robust_pareto.json")
        if isinstance(robust_pareto_payload, list):
            robust_pareto = robust_pareto_payload[:max_pareto]
            evidence.append(store.recorder.relpath(run_dir / "search" / "robust_pareto.json"))

        sensitivity_payload = _read_json(run_dir / "insights" / "sensitivity.json")
        if isinstance(sensitivity_payload, Mapping):
            sensitivity = dict(sensitivity_payload)
            evidence.append(store.recorder.relpath(run_dir / "insights" / "sensitivity.json"))

    # Latest spec/cfg payloads are stored as rev snapshots (deterministic).
    spec_path = session_dir / "spec_revs" / f"spec_rev{state.spec_rev:04d}.json"
    cfg_path = session_dir / "cfg_revs" / f"cfg_rev{state.cfg_rev:04d}.json"
    spec_payload = _read_json(spec_path) or {}
    cfg_payload = _read_json(cfg_path) or {}
    evidence.append(store.recorder.relpath(spec_path))
    evidence.append(store.recorder.relpath(cfg_path))

    return {
        "session": {
            "session_id": state.session_id,
            "bench_id": state.bench_id,
            "seed": state.seed,
            "created_at": state.created_at,
            "current_phase": state.current_phase,
            "spec_rev": state.spec_rev,
            "cfg_rev": state.cfg_rev,
            "spec_hash": state.spec_hash,
            "cfg_hash": state.cfg_hash,
        },
        "limits": {
            "max_topk": max_topk,
            "max_pareto": max_pareto,
            "max_report_section_lines": max_report_section_lines,
        },
        "meta_report_excerpt": _clip_lines(meta_report, max_lines=120),
        "grid_run_dir": grid_run_dir,
        "grid_report_sections": report_sections,
        "topk": topk,
        "pareto": pareto,
        "robust_topk": robust_topk,
        "robust_pareto": robust_pareto,
        "sensitivity": sensitivity,
        "spec": spec_payload,
        "cfg": cfg_payload,
        "evidence": sorted(set(evidence)),
    }

