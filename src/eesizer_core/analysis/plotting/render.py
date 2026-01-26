from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        return plt, np
    except Exception:
        return None, None


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def render_heatmap(data: Mapping[str, Any], out_path: Path) -> tuple[bool, str | None]:
    plt, np = _import_matplotlib()
    if plt is None or np is None:
        return False, "matplotlib_missing"

    matrix = data.get("matrix") or []
    if not matrix:
        return False, "no_rows"

    param_ids = data.get("param_ids") or []
    row_labels = data.get("row_labels") or []

    arr = np.array(
        [
            [float(val) if val is not None else float("nan") for val in row]
            for row in matrix
        ],
        dtype=float,
    )

    fig_w = max(6.0, 0.5 * len(param_ids))
    fig_h = max(3.0, 0.3 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    cmap = plt.cm.coolwarm
    cmap.set_bad(color="#dddddd")

    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(param_ids)))
    ax.set_xticklabels(param_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Knob Delta Heatmap (log10(value/nominal))")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, None


def render_scatter(
    data: Mapping[str, Any],
    out_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    pm_target: float | None = None,
) -> tuple[bool, str | None]:
    plt, np = _import_matplotlib()
    if plt is None or np is None:
        return False, "matplotlib_missing"

    points = data.get("points") or []
    if not points:
        return False, "no_points"

    status_colors = {
        "ok": "#1f77b4",
        "objective_fail": "#8c8c8c",
        "metric_missing": "#ff7f0e",
        "sim_fail": "#d62728",
        "guard_fail": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    for status, color in status_colors.items():
        xs = [p.get("x") for p in points if p.get("status") == status]
        ys = [p.get("y") for p in points if p.get("status") == status]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if not xs or not ys:
            continue
        ax.scatter(xs, ys, s=30, alpha=0.8, label=status, color=color)

    # Highlight pareto/topk/best
    highlight_tags = {"pareto": "#2ca02c", "topk": "#17becf", "best": "#e377c2"}
    for tag, color in highlight_tags.items():
        xs = [p.get("x") for p in points if tag in (p.get("tags") or [])]
        ys = [p.get("y") for p in points if tag in (p.get("tags") or [])]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if not xs or not ys:
            continue
        ax.scatter(xs, ys, s=80, facecolors="none", edgecolors=color, linewidths=1.5, label=tag)

    if pm_target is not None:
        ax.axhline(pm_target, color="#555555", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, None


def render_failure_breakdown(data: Mapping[str, Any], out_path: Path) -> tuple[bool, str | None]:
    plt, np = _import_matplotlib()
    if plt is None or np is None:
        return False, "matplotlib_missing"

    counts = data.get("counts") or {}
    if not counts:
        return False, "no_counts"

    labels = ["guard_fail", "sim_fail", "metric_missing", "objective_fail", "ok"]
    values = [int(counts.get(label, 0)) for label in labels]

    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=150)
    positions = list(range(len(labels)))
    ax.bar(positions, values, color="#4c72b0")
    ax.set_title("Failure Breakdown")
    ax.set_ylabel("count")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, None


def render_nominal_vs_worst(data: Mapping[str, Any], out_path: Path) -> tuple[bool, str | None]:
    plt, np = _import_matplotlib()
    if plt is None or np is None:
        return False, "matplotlib_missing"

    points = data.get("points") or []
    if not points:
        return False, "no_points"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=150)

    for ax, metric, title in zip(
        axes,
        ("ugbw_hz", "power_w"),
        ("Nominal vs Worst UGBW", "Nominal vs Worst Power"),
    ):
        xs = [p.get("nominal", {}).get(metric) for p in points]
        ys = [p.get("worst", {}).get(metric) for p in points]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if not xs or not ys:
            ax.set_visible(False)
            continue
        ax.scatter(xs, ys, s=40, color="#1f77b4", alpha=0.8)
        min_val = min(xs + ys)
        max_val = max(xs + ys)
        ax.plot([min_val, max_val], [min_val, max_val], color="#555555", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(f"nominal {metric}")
        ax.set_ylabel(f"worst {metric}")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, None
