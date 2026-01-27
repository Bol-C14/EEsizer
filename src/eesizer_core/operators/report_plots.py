from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..analysis.plotting import (
    PlotEntry,
    build_plot_index,
    extract_plot_context,
    build_heatmap_data,
    build_tradeoff_data,
    build_pm_vs_ugbw_data,
    build_failure_breakdown_data,
    build_nominal_vs_worst_data,
    render_heatmap,
    render_scatter,
    render_failure_breakdown,
    render_nominal_vs_worst,
    build_sensitivity_insights,
)
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ..runtime.recorder import RunRecorder


@dataclass(frozen=True)
class PlotPaths:
    png: str
    data: str


class ReportPlotsOperator(Operator):
    """Generate deterministic plot data + PNGs and update report."""

    name = "report_plots"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        run_dir = inputs.get("run_dir")
        recorder = inputs.get("recorder")
        manifest = inputs.get("manifest")
        report_path = inputs.get("report_path", "report.md")

        if recorder is None:
            if run_dir is None:
                raise ValueError("ReportPlotsOperator requires run_dir or recorder")
            recorder = RunRecorder(Path(run_dir))

        run_dir = recorder.run_dir
        report_file = run_dir / report_path
        report_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""

        context = extract_plot_context(run_dir)

        plots: dict[str, PlotPaths] = {
            "knob_delta_heatmap": PlotPaths(
                png="plots/knob_delta_heatmap.png",
                data="plots/knob_delta_heatmap_data.json",
            ),
            "tradeoff_power_vs_ugbw": PlotPaths(
                png="plots/tradeoff_power_vs_ugbw.png",
                data="plots/tradeoff_power_vs_ugbw_data.json",
            ),
            "tradeoff_pm_vs_ugbw": PlotPaths(
                png="plots/tradeoff_pm_vs_ugbw.png",
                data="plots/tradeoff_pm_vs_ugbw_data.json",
            ),
            "tradeoff_objectives": PlotPaths(
                png="plots/tradeoff_objectives.png",
                data="plots/tradeoff_objectives_data.json",
            ),
            "failures_breakdown": PlotPaths(
                png="plots/failures_breakdown.png",
                data="plots/failures_breakdown_data.json",
            ),
            "robust_nominal_vs_worst": PlotPaths(
                png="plots/robust_nominal_vs_worst.png",
                data="plots/robust_nominal_vs_worst_data.json",
            ),
        }

        plot_entries: list[PlotEntry] = []
        skipped: list[tuple[str, str]] = []

        # Heatmap
        heatmap_data = build_heatmap_data(context.rows, context.param_ids)
        heatmap_sha = stable_hash_json(heatmap_data)
        recorder.write_json(plots["knob_delta_heatmap"].data, heatmap_data)
        if heatmap_data.get("matrix") and heatmap_data.get("param_ids"):
            ok, reason = render_heatmap(heatmap_data, run_dir / plots["knob_delta_heatmap"].png)
        else:
            ok, reason = False, "no_data"
        plot_entries.append(
            PlotEntry(
                name="knob_delta_heatmap",
                png_path=plots["knob_delta_heatmap"].png if ok else None,
                data_path=plots["knob_delta_heatmap"].data,
                data_sha256=heatmap_sha,
                status="ok" if ok else "skipped",
                skip_reason=reason,
            )
        )
        if not ok and reason:
            skipped.append(("knob_delta_heatmap", reason))

        # Tradeoff Power vs UGBW
        tradeoff_data = build_tradeoff_data(context.rows, x_metric="power_w", y_metric="ugbw_hz")
        tradeoff_sha = stable_hash_json(tradeoff_data)
        recorder.write_json(plots["tradeoff_power_vs_ugbw"].data, tradeoff_data)
        if _has_points(tradeoff_data):
            ok, reason = render_scatter(
                tradeoff_data,
                run_dir / plots["tradeoff_power_vs_ugbw"].png,
                title="Power vs UGBW",
                x_label="power_w",
                y_label="ugbw_hz",
            )
        else:
            ok, reason = False, "no_points"
        plot_entries.append(
            PlotEntry(
                name="tradeoff_power_vs_ugbw",
                png_path=plots["tradeoff_power_vs_ugbw"].png if ok else None,
                data_path=plots["tradeoff_power_vs_ugbw"].data,
                data_sha256=tradeoff_sha,
                status="ok" if ok else "skipped",
                skip_reason=reason,
            )
        )
        if not ok and reason:
            skipped.append(("tradeoff_power_vs_ugbw", reason))

        # Fallback: tradeoff between primary objectives (e.g., RC)
        if not ok and reason == "no_points":
            axes = _select_objective_tradeoff_axes(context.spec_payload)
            if axes is None:
                obj_ok, obj_reason = False, "no_objective_pair"
                obj_data: dict[str, Any] = {
                    "plot": "tradeoff_objectives",
                    "x_metric": None,
                    "y_metric": None,
                    "points": [],
                }
            else:
                x_metric, y_metric = axes
                obj_data = build_tradeoff_data(context.rows, x_metric=x_metric, y_metric=y_metric)
                if _has_points(obj_data):
                    obj_ok, obj_reason = render_scatter(
                        obj_data,
                        run_dir / plots["tradeoff_objectives"].png,
                        title=f"Tradeoff: {y_metric} vs {x_metric}",
                        x_label=x_metric,
                        y_label=y_metric,
                    )
                else:
                    obj_ok, obj_reason = False, "no_points"

            obj_sha = stable_hash_json(obj_data)
            recorder.write_json(plots["tradeoff_objectives"].data, obj_data)
            plot_entries.append(
                PlotEntry(
                    name="tradeoff_objectives",
                    png_path=plots["tradeoff_objectives"].png if obj_ok else None,
                    data_path=plots["tradeoff_objectives"].data,
                    data_sha256=obj_sha,
                    status="ok" if obj_ok else "skipped",
                    skip_reason=obj_reason,
                    notes={
                        "x_metric": obj_data.get("x_metric"),
                        "y_metric": obj_data.get("y_metric"),
                    },
                )
            )
            if not obj_ok and obj_reason:
                skipped.append(("tradeoff_objectives", obj_reason))

        # PM vs UGBW
        pm_data = build_pm_vs_ugbw_data(context.rows, context.spec_payload)
        pm_sha = stable_hash_json(pm_data)
        recorder.write_json(plots["tradeoff_pm_vs_ugbw"].data, pm_data)
        if _has_points(pm_data):
            ok, reason = render_scatter(
                pm_data,
                run_dir / plots["tradeoff_pm_vs_ugbw"].png,
                title="Phase Margin vs UGBW",
                x_label="ugbw_hz",
                y_label="phase_margin_deg",
                pm_target=pm_data.get("pm_target"),
            )
        else:
            ok, reason = False, "no_points"
        plot_entries.append(
            PlotEntry(
                name="tradeoff_pm_vs_ugbw",
                png_path=plots["tradeoff_pm_vs_ugbw"].png if ok else None,
                data_path=plots["tradeoff_pm_vs_ugbw"].data,
                data_sha256=pm_sha,
                status="ok" if ok else "skipped",
                skip_reason=reason,
            )
        )
        if not ok and reason:
            skipped.append(("tradeoff_pm_vs_ugbw", reason))

        # Failure breakdown
        failure_data = build_failure_breakdown_data(context.rows)
        failure_sha = stable_hash_json(failure_data)
        recorder.write_json(plots["failures_breakdown"].data, failure_data)
        if _has_counts(failure_data):
            ok, reason = render_failure_breakdown(failure_data, run_dir / plots["failures_breakdown"].png)
        else:
            ok, reason = False, "no_counts"
        plot_entries.append(
            PlotEntry(
                name="failures_breakdown",
                png_path=plots["failures_breakdown"].png if ok else None,
                data_path=plots["failures_breakdown"].data,
                data_sha256=failure_sha,
                status="ok" if ok else "skipped",
                skip_reason=reason,
            )
        )
        if not ok and reason:
            skipped.append(("failures_breakdown", reason))

        # Robust nominal vs worst (corner runs)
        robust_data = build_nominal_vs_worst_data(context.robust_rows)
        robust_sha = stable_hash_json(robust_data)
        recorder.write_json(plots["robust_nominal_vs_worst"].data, robust_data)
        if _has_points(robust_data):
            ok, reason = render_nominal_vs_worst(robust_data, run_dir / plots["robust_nominal_vs_worst"].png)
        else:
            ok, reason = False, "no_points"
        plot_entries.append(
            PlotEntry(
                name="robust_nominal_vs_worst",
                png_path=plots["robust_nominal_vs_worst"].png if ok else None,
                data_path=plots["robust_nominal_vs_worst"].data,
                data_sha256=robust_sha,
                status="ok" if ok else "skipped",
                skip_reason=reason,
            )
        )
        if not ok and reason:
            skipped.append(("robust_nominal_vs_worst", reason))

        index_payload = build_plot_index(plot_entries)
        recorder.write_json("plots/index.json", index_payload)

        # Insights
        metric_names = ["ugbw_hz", "phase_margin_deg", "power_w", "score"]
        insights = build_sensitivity_insights(context.rows, context.param_ids, metric_names)
        insights_path = None
        if insights.get("top_params"):
            insights_path = recorder.write_json("insights/sensitivity.json", insights)

        updated_report = _update_report(report_text, plot_entries, insights)
        recorder.write_text(report_path, updated_report)

        if manifest is not None:
            manifest.files.setdefault("plots/index.json", "plots/index.json")
            for entry in plot_entries:
                if entry.data_path:
                    manifest.files.setdefault(entry.data_path, entry.data_path)
                if entry.png_path:
                    manifest.files.setdefault(entry.png_path, entry.png_path)
            if insights_path is not None:
                manifest.files.setdefault("insights/sensitivity.json", "insights/sensitivity.json")

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["run_dir"] = ArtifactFingerprint(sha256=stable_hash_str(str(run_dir)))
        prov.outputs["plots_index"] = ArtifactFingerprint(sha256=stable_hash_json(index_payload))
        prov.finish()

        return OperatorResult(
            outputs={
                "plots_index": index_payload,
                "plots_skipped": skipped,
                "insights": insights,
            },
            provenance=prov,
        )


def _select_objective_tradeoff_axes(spec_payload: Mapping[str, Any]) -> tuple[str, str] | None:
    objectives = spec_payload.get("objectives", []) if isinstance(spec_payload, Mapping) else []
    metric_ids: list[str] = []
    for obj in objectives or []:
        if not isinstance(obj, Mapping):
            continue
        metric = obj.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            continue
        mid = metric.strip()
        if mid not in metric_ids:
            metric_ids.append(mid)
    if len(metric_ids) < 2:
        return None

    # Heuristic: prefer x=dc_* and y=ac_* for RC-like specs.
    ac = next((m for m in metric_ids if m.lower().startswith("ac_") or "ac_" in m.lower()), None)
    dc = next((m for m in metric_ids if m.lower().startswith("dc_") or "dc_" in m.lower()), None)
    if ac and dc and ac != dc:
        return dc, ac
    return metric_ids[0], metric_ids[1]


def _has_points(data: Mapping[str, Any]) -> bool:
    points = data.get("points") or []
    if not points:
        return False
    for point in points:
        x = point.get("x")
        y = point.get("y")
        if x is not None and y is not None:
            return True
        if "nominal" in point and "worst" in point:
            nominal = point.get("nominal") or {}
            worst = point.get("worst") or {}
            for metric in ("ugbw_hz", "power_w"):
                if nominal.get(metric) is not None and worst.get(metric) is not None:
                    return True
    return False


def _has_counts(data: Mapping[str, Any]) -> bool:
    counts = data.get("counts") or {}
    return any(int(v) > 0 for v in counts.values())


def _update_report(report_text: str, plot_entries: list[PlotEntry], insights: Mapping[str, Any]) -> str:
    lines = report_text.splitlines()
    plot_section = _build_plot_section(plot_entries)
    lines = _replace_section(lines, "## Plots", plot_section)

    insight_section = _build_insight_section(insights)
    if insight_section:
        lines = _replace_section(lines, "## Insights", insight_section)
    return "\n".join(lines).rstrip() + "\n"


def _build_plot_section(plot_entries: list[PlotEntry]) -> list[str]:
    lines: list[str] = ["## Plots", ""]
    for entry in plot_entries:
        if entry.status == "ok" and entry.png_path:
            lines.append(f"![]({entry.png_path})")
            lines.append("")
    for entry in plot_entries:
        if entry.status != "ok":
            reason = entry.skip_reason or "skipped"
            lines.append(f"- {entry.name}: skipped ({reason})")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _build_insight_section(insights: Mapping[str, Any]) -> list[str]:
    top_params = insights.get("top_params") if isinstance(insights, Mapping) else None
    if not isinstance(top_params, Mapping) or not top_params:
        return []
    lines = ["## Insights", "", "Top parameter sensitivities (Spearman):"]
    for metric in sorted(top_params.keys()):
        items = top_params.get(metric) or []
        parts = []
        for item in items:
            param_id = item.get("param_id")
            corr = item.get("spearman")
            if param_id is None or corr is None:
                continue
            parts.append(f"{param_id} ({corr:+.3f})")
        if parts:
            lines.append(f"- {metric}: " + ", ".join(parts))
    return lines


def _replace_section(lines: list[str], heading: str, new_section: list[str]) -> list[str]:
    if not new_section:
        return lines
    try:
        start = lines.index(heading)
    except ValueError:
        if lines and lines[-1].strip():
            lines.append("")
        return lines + new_section

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return lines[:start] + new_section + lines[end:]
