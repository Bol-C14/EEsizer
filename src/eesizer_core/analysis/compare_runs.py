from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import CircuitSpec, MetricsBundle, MetricValue, Objective
from ..metrics.aliases import canonicalize_metrics, canonicalize_metric_name
from ..metrics.tolerances import DEFAULT_TOL
from ..runtime.run_loader import RunLoader
from ..strategies.objective_eval import evaluate_objectives


_EPS = 1e-12


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    manifest: dict[str, Any]
    spec_payload: dict[str, Any]
    metrics_payload: dict[str, Any]
    summary_payload: dict[str, Any]
    history_entries: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spec_payload(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "inputs" / "spec.json")


def _load_summary_payload(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "history" / "summary.json")


def _build_spec(spec_payload: Mapping[str, Any]) -> CircuitSpec:
    objectives = []
    for obj in spec_payload.get("objectives", []) or []:
        if not isinstance(obj, Mapping):
            continue
        metric = obj.get("metric")
        if not metric:
            continue
        objectives.append(
            Objective(
                metric=canonicalize_metric_name(str(metric)),
                target=obj.get("target"),
                tol=obj.get("tol"),
                weight=obj.get("weight", 1.0),
                sense=obj.get("sense", "ge"),
            )
        )
    return CircuitSpec(objectives=tuple(objectives))


def _build_metrics_bundle(metrics_payload: Mapping[str, Any]) -> MetricsBundle:
    mb = MetricsBundle()
    for name, value in metrics_payload.items():
        if isinstance(value, Mapping):
            details = value.get("details")
            details_payload = dict(details) if isinstance(details, Mapping) else {}
            mv = MetricValue(
                name=name,
                value=value.get("value"),
                unit=value.get("unit", ""),
                passed=value.get("passed"),
                details=details_payload,
            )
        else:
            mv = MetricValue(name=name, value=value, unit="")
        mb.values[name] = mv
    return mb


def _metric_value(payload: Mapping[str, Any], name: str) -> tuple[float | None, str]:
    value = payload.get(name)
    if isinstance(value, Mapping):
        return value.get("value"), value.get("unit", "")
    if isinstance(value, (int, float)):
        return float(value), ""
    return None, ""


def _parse_iso_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _wall_time_s(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_iso_time(start)
    end_dt = _parse_iso_time(end)
    if start_dt is None or end_dt is None:
        return None
    delta = end_dt - start_dt
    return max(0.0, delta.total_seconds())


def _resolve_tolerances(spec_payload: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    tolerances = {name: dict(cfg) for name, cfg in DEFAULT_TOL.items()}
    notes = spec_payload.get("notes", {})
    if isinstance(notes, Mapping):
        overrides = notes.get("compare_tol")
        if isinstance(overrides, Mapping):
            for key, cfg in overrides.items():
                if not isinstance(cfg, Mapping):
                    continue
                tolerances[str(key)] = {
                    "abs": cfg.get("abs"),
                    "rel": cfg.get("rel"),
                }
    return tolerances


def _collect_history(run_dir: Path) -> list[dict[str, Any]]:
    loader = RunLoader(run_dir)
    return list(loader.iter_history())


def _load_snapshot(run_dir: Path) -> RunSnapshot:
    run_dir = Path(run_dir)
    loader = RunLoader(run_dir)
    manifest = loader.load_manifest()
    spec_payload = _load_spec_payload(run_dir)
    best = loader.load_best()
    metrics_payload = best.get("best_metrics", {}) if isinstance(best, dict) else {}
    summary_payload = _load_summary_payload(run_dir)
    history_entries = _collect_history(run_dir)
    run_id = manifest.get("run_id") or run_dir.name
    return RunSnapshot(
        run_id=run_id,
        manifest=manifest,
        spec_payload=spec_payload,
        metrics_payload=metrics_payload,
        summary_payload=summary_payload,
        history_entries=history_entries,
    )


def _count_attempts(history: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for entry in history:
        attempts = entry.get("attempts", [])
        if isinstance(attempts, list):
            count += len(attempts)
    return count


def _build_metric_rows(
    metrics_a: Mapping[str, Any],
    metrics_b: Mapping[str, Any],
    tolerances: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    keys = set(metrics_a.keys()) | set(metrics_b.keys())
    for name in sorted(keys):
        a_val, a_unit = _metric_value(metrics_a, name)
        b_val, b_unit = _metric_value(metrics_b, name)
        delta = None if a_val is None or b_val is None else b_val - a_val
        rel_delta = None
        if delta is not None and a_val is not None and abs(a_val) > _EPS:
            rel_delta = delta / abs(a_val)
        tol_cfg = tolerances.get(name, {})
        abs_tol = tol_cfg.get("abs")
        rel_tol = tol_cfg.get("rel")
        within_tol = None
        if delta is not None and (abs_tol is not None or rel_tol is not None):
            within_tol = False
            if abs_tol is not None and abs(delta) <= abs_tol:
                within_tol = True
            if rel_tol is not None and rel_delta is not None and abs(rel_delta) <= rel_tol:
                within_tol = True
        rows[name] = {
            "a": a_val,
            "b": b_val,
            "delta": delta,
            "rel_delta": rel_delta,
            "unit": a_unit or b_unit,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "within_tol": within_tol,
        }
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1e3 or (abs(value) > 0 and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _objective_rows(spec: CircuitSpec, metrics_bundle: MetricsBundle) -> list[dict[str, Any]]:
    report = evaluate_objectives(spec, metrics_bundle)
    return report["per_objective"]


def _objectives_match(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> bool:
    if len(rows_a) != len(rows_b):
        return False
    for a, b in zip(rows_a, rows_b):
        if a.get("metric") != b.get("metric"):
            return False
        if a.get("passed") != b.get("passed"):
            return False
    return True


def compare_runs(run_dir_a: Path, run_dir_b: Path, out_dir: Path) -> dict[str, Any]:
    snap_a = _load_snapshot(run_dir_a)
    snap_b = _load_snapshot(run_dir_b)
    spec_payload = snap_a.spec_payload or snap_b.spec_payload
    spec = _build_spec(spec_payload)

    metrics_a = canonicalize_metrics(snap_a.metrics_payload)
    metrics_b = canonicalize_metrics(snap_b.metrics_payload)
    tolerances = _resolve_tolerances(spec_payload)

    metrics_bundle_a = _build_metrics_bundle(metrics_a)
    metrics_bundle_b = _build_metrics_bundle(metrics_b)
    objectives_a = _objective_rows(spec, metrics_bundle_a)
    objectives_b = _objective_rows(spec, metrics_bundle_b)
    objectives_match = _objectives_match(objectives_a, objectives_b)

    metric_rows = _build_metric_rows(metrics_a, metrics_b, tolerances)
    within_fail = any(row.get("within_tol") is False for row in metric_rows.values())

    manifest_a = snap_a.manifest
    manifest_b = snap_b.manifest
    summary_a = snap_a.summary_payload
    summary_b = snap_b.summary_payload
    history_a = snap_a.history_entries
    history_b = snap_b.history_entries

    comparison = {
        "run_a": {
            "run_id": snap_a.run_id,
            "timestamp_start": manifest_a.get("timestamp_start"),
            "timestamp_end": manifest_a.get("timestamp_end"),
            "sim_runs_total": summary_a.get("sim_runs_total"),
            "iterations": len(history_a),
            "attempts": _count_attempts(history_a),
            "wall_time_s": _wall_time_s(manifest_a.get("timestamp_start"), manifest_a.get("timestamp_end")),
            "stop_reason": summary_a.get("stop_reason"),
        },
        "run_b": {
            "run_id": snap_b.run_id,
            "timestamp_start": manifest_b.get("timestamp_start"),
            "timestamp_end": manifest_b.get("timestamp_end"),
            "sim_runs_total": summary_b.get("sim_runs_total"),
            "iterations": len(history_b),
            "attempts": _count_attempts(history_b),
            "wall_time_s": _wall_time_s(manifest_b.get("timestamp_start"), manifest_b.get("timestamp_end")),
            "stop_reason": summary_b.get("stop_reason"),
        },
        "objectives": {
            "run_a": objectives_a,
            "run_b": objectives_b,
            "match": objectives_match,
        },
        "metrics": metric_rows,
        "equivalent": (not within_fail) and objectives_match,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Run Comparison")
    lines.append("")
    lines.append("## Run A / Run B")
    lines.append("")
    lines.append("| Field | Run A | Run B |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| run_id | {_fmt(comparison['run_a']['run_id'])} | {_fmt(comparison['run_b']['run_id'])} |")
    lines.append(f"| timestamp_start | {_fmt(comparison['run_a']['timestamp_start'])} | {_fmt(comparison['run_b']['timestamp_start'])} |")
    lines.append(f"| timestamp_end | {_fmt(comparison['run_a']['timestamp_end'])} | {_fmt(comparison['run_b']['timestamp_end'])} |")
    lines.append(f"| sim_runs_total | {_fmt(comparison['run_a']['sim_runs_total'])} | {_fmt(comparison['run_b']['sim_runs_total'])} |")
    lines.append(f"| iterations | {_fmt(comparison['run_a']['iterations'])} | {_fmt(comparison['run_b']['iterations'])} |")
    lines.append(f"| attempts | {_fmt(comparison['run_a']['attempts'])} | {_fmt(comparison['run_b']['attempts'])} |")
    lines.append(f"| wall_time_s | {_fmt(comparison['run_a']['wall_time_s'])} | {_fmt(comparison['run_b']['wall_time_s'])} |")
    lines.append(f"| stop_reason | {_fmt(comparison['run_a']['stop_reason'])} | {_fmt(comparison['run_b']['stop_reason'])} |")
    lines.append("")
    lines.append("## Objectives Pass/Fail")
    lines.append("")
    lines.append("| Metric | Run A | Run B |")
    lines.append("| --- | --- | --- |")
    for obj_a, obj_b in zip(objectives_a, objectives_b):
        lines.append(f"| {obj_a.get('metric')} | {_fmt(obj_a.get('passed'))} | {_fmt(obj_b.get('passed'))} |")
    lines.append("")
    lines.append("## Metrics Diff")
    lines.append("")
    lines.append("| Metric | A | B | Delta | Rel | Abs Tol | Rel Tol | Within |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, row in metric_rows.items():
        lines.append(
            "| {metric} | {a} | {b} | {delta} | {rel} | {abs_tol} | {rel_tol} | {within} |".format(
                metric=name,
                a=_fmt(row.get("a")),
                b=_fmt(row.get("b")),
                delta=_fmt(row.get("delta")),
                rel=_fmt(row.get("rel_delta")),
                abs_tol=_fmt(row.get("abs_tol")),
                rel_tol=_fmt(row.get("rel_tol")),
                within=_fmt(row.get("within_tol")),
            )
        )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if comparison["equivalent"]:
        lines.append("Runs are equivalent within tolerance and objective pass/fail matches.")
    else:
        lines.append("Runs differ beyond tolerance and/or objective pass/fail mismatches.")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def _default_out_dir(run_dir_a: Path, run_dir_b: Path) -> Path:
    root = Path.cwd() / "comparisons"
    root.mkdir(parents=True, exist_ok=True)
    name = f"{Path(run_dir_a).name}_vs_{Path(run_dir_b).name}"
    return root / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two run directories.")
    parser.add_argument("--run-a", required=True, type=Path, help="Path to run A directory.")
    parser.add_argument("--run-b", required=True, type=Path, help="Path to run B directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for comparison artifacts.")
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else _default_out_dir(args.run_a, args.run_b)
    compare_runs(args.run_a, args.run_b, out_dir)
    print(f"Comparison written to {out_dir}")


if __name__ == "__main__":
    main()
