"""Reporting helpers to serialize optimization history and summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from ..context import ArtifactKind, ExecutionContext


class OptimizationReporter:
    """Writes optimization summaries and artifacts in a reusable way."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

    def write_summary(self, context: ExecutionContext, metrics: Mapping[str, float]) -> Path:
        path = self.artifacts_dir / "optimization_summary.json"
        path.write_text(json.dumps(metrics, indent=2))
        context.attach_artifact(
            "optimization_summary",
            path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Notebook-style optimization summary containing iterations and flags.",
        )
        return path

    def write_history(
        self,
        context: ExecutionContext,
        history: Sequence[Mapping[str, object]],
    ) -> tuple[Path, Path, Path]:
        csv_path = self.artifacts_dir / "optimization_history.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["iteration", "gain_db", "power_mw", "analysis_note"]
            )
            writer.writeheader()
            for entry in history:
                writer.writerow(
                    {
                        "iteration": f"{entry.get('iteration', 0):.0f}",
                        "gain_db": f"{entry.get('gain_db', 0.0):.3f}",
                        "power_mw": f"{entry.get('power_mw', 0.0):.3f}",
                        "analysis_note": entry.get("analysis_note", ""),
                    }
                )
        context.attach_artifact(
            "optimization_history_csv",
            csv_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Iteration-by-iteration metrics stored as CSV.",
        )

        log_path = self.artifacts_dir / "optimization_history.log"
        history_text = []
        for entry in history:
            history_text.append(
                f"Iteration {entry.get('iteration', 0):.0f}: gain={entry.get('gain_db', 0.0):.3f} dB, power={entry.get('power_mw', 0.0):.3f} mW\n"
                f"{entry.get('analysis_note','')}\n{entry.get('optimization_note','')}\n{entry.get('sizing_note','')}"
            )
        log_path.write_text("\n\n".join(history_text))
        context.attach_artifact(
            "optimization_history_log",
            log_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Text log summarizing prompts and adjustments per iteration.",
        )

        pdf_path = self.artifacts_dir / "optimization_history.pdf"
        pdf_path.write_text("Notebook-equivalent optimization report\n\n" + "\n\n".join(history_text))
        context.attach_artifact(
            "optimization_history_pdf",
            pdf_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Lightweight PDF placeholder mirroring the notebook artifact list.",
        )
        return csv_path, log_path, pdf_path

    def write_variant_comparison(
        self,
        context: ExecutionContext,
        variants: Sequence[tuple[str, Mapping[str, float]]],
        *,
        scoring_fn: Callable[[Mapping[str, float]], float] | None = None,
    ) -> tuple[Path, Path]:
        """Persist multi-variant scores to CSV/JSON for best-of-N selection."""

        entries = []
        for name, metrics in variants:
            score = scoring_fn(metrics) if scoring_fn else 0.0
            entries.append({"name": name, "score": float(score), "metrics": dict(metrics)})
        entries.sort(key=lambda item: item["score"], reverse=True)

        csv_path = self.artifacts_dir / "variant_comparison.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "score", "metrics"])
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "name": entry["name"],
                        "score": f"{entry['score']:.4f}",
                        "metrics": json.dumps(entry["metrics"]),
                    }
                )
        context.attach_artifact(
            "variant_comparison_csv",
            csv_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Comparison of candidate netlist variants with scores.",
        )

        json_path = self.artifacts_dir / "variant_comparison.json"
        json_path.write_text(json.dumps(entries, indent=2))
        context.attach_artifact(
            "variant_comparison_json",
            json_path,
            kind=ArtifactKind.OPTIMIZATION,
            description="Comparison of candidate netlist variants with scores (JSON).",
        )
        return csv_path, json_path


__all__ = ["OptimizationReporter"]
