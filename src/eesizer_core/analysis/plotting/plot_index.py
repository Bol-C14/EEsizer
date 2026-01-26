from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class PlotEntry:
    name: str
    png_path: str | None
    data_path: str | None
    data_sha256: str | None
    status: str
    skip_reason: str | None = None
    created_by: str = "report_plots"
    versions: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "png_path": self.png_path,
            "data_path": self.data_path,
            "data_sha256": self.data_sha256,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "created_by": self.created_by,
            "versions": dict(self.versions),
            "notes": dict(self.notes),
        }
        return payload


def build_plot_index(entries: Iterable[PlotEntry]) -> dict[str, Any]:
    sorted_entries = sorted(entries, key=lambda e: e.name)
    return {
        "plots": [entry.to_dict() for entry in sorted_entries],
    }
