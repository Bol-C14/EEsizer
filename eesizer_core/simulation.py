"""Simulation helpers used by agents and tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from math import log10
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, MutableMapping, Sequence

from .config import SimulationConfig
from .netlist import summarize_components
from .spice import ControlDeck, augment_netlist, load_measurements, parse_measure_log


@dataclass(slots=True)
class SimulationResult:
    metrics: MutableMapping[str, float]


class MockNgSpiceSimulator:
    """Deterministic, dependency-free simulator stub."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(
        self,
        netlist_text: str,
        deck: ControlDeck | None = None,
        workdir: Path | None = None,
    ) -> MutableMapping[str, float]:
        counts = summarize_components(netlist_text)
        mos = counts.get("M", 0)
        resistors = counts.get("R", 0)
        capacitors = counts.get("C", 0)
        diodes = counts.get("D", 0)

        gain_db = 20.0 + mos * 4.5
        power_mw = max(0.1, 0.4 + resistors * 0.15 + mos * 0.05)
        bandwidth_hz = 1e6 / (1 + capacitors)
        noise_mv = 0.25 + max(0.0, 2.0 - log10(1 + mos + diodes))

        return {
            "gain_db": gain_db,
            "power_mw": round(power_mw, 3),
            "bandwidth_hz": round(bandwidth_hz, 3),
            "noise_mv": round(noise_mv, 3),
            "transistor_count": float(mos),
        }


INCLUDE_PATTERN = re.compile(r"^\s*\.include\s+['\"]?(?P<path>[^'\"\s]+)['\"]?", re.IGNORECASE)


class NgSpiceRunner:
    """Invoke a real ngspice binary with generated control decks."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        include_paths: Sequence[Path] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        discovered = list(include_paths or [])
        default_root = Path(__file__).resolve().parents[1]
        for candidate in (
            default_root / "agent_test_gpt",
            default_root / "agent_test_gemini",
            default_root / "variation",
        ):
            if candidate.exists():
                discovered.append(candidate)
        self.include_paths = tuple(Path(path).resolve() for path in discovered)
        self.environment = dict(environment or {})

    def run(
        self,
        netlist_text: str,
        deck: ControlDeck | None = None,
        workdir: Path | None = None,
    ) -> MutableMapping[str, float]:
        augmented = augment_netlist(netlist_text, deck) if deck else netlist_text
        temp_dir_obj = None
        workspace: Path
        if workdir:
            workspace = Path(workdir)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir_obj = tempfile.TemporaryDirectory()
            workspace = Path(temp_dir_obj.name)

        netlist_path = workspace / "circuit.cir"
        netlist_path.write_text(augmented)
        self._materialize_includes(augmented, workspace)
        log_path = workspace / "ngspice.log"

        command = [str(self.config.binary_path), "-b", "-o", str(log_path), str(netlist_path)]
        env = os.environ.copy()
        env.update(self.environment)
        try:
            subprocess.run(
                command,
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )
            log_text = log_path.read_text() if log_path.exists() else ""
            metrics = parse_measure_log(log_text)
            metrics.update(self._collect_measurement_files(workspace))
            return metrics
        except FileNotFoundError as exc:
            raise RuntimeError(f"ngspice binary not found: {self.config.binary_path}") from exc
        finally:
            if temp_dir_obj:
                temp_dir_obj.cleanup()

    def _collect_measurement_files(self, workspace: Path) -> MutableMapping[str, float]:
        """Load any measurement artifacts ngspice left behind in the workspace."""

        aggregated: MutableMapping[str, float] = {}
        for candidate in workspace.glob("*.measure"):
            try:
                aggregated.update(load_measurements(candidate))
            except OSError:
                continue
        for candidate in (workspace / "measurements.json", workspace / "metrics.json"):
            if candidate.exists():
                try:
                    aggregated.update(load_measurements(candidate))
                except OSError:
                    continue
        return aggregated

    def _materialize_includes(self, text: str, workspace: Path) -> None:
        for line in text.splitlines():
            match = INCLUDE_PATTERN.match(line)
            if not match:
                continue
            target = match.group("path")
            include_path = Path(target)
            if include_path.is_absolute():
                continue
            if (workspace / include_path).exists():
                continue
            resolved = self._find_include(include_path)
            if resolved:
                destination = workspace / include_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved, destination)

    def _find_include(self, path: Path) -> Path | None:
        for base in self.include_paths:
            candidate = (base / path) if not path.is_absolute() else path
            if candidate.exists():
                return candidate
        return None


__all__ = ["MockNgSpiceSimulator", "NgSpiceRunner", "SimulationResult"]
