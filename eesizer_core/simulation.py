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
from typing import Iterable, Mapping, MutableMapping, Sequence

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
        artifact_dir: Path | None = None,
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
        artifact_dir: Path | None = None,
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
            wave_metrics, artifacts = self._collect_waveform_metrics(workspace)
            metrics.update(wave_metrics)
            if artifact_dir:
                self._persist_artifacts(artifacts, artifact_dir)
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

    def _collect_waveform_metrics(
        self, workspace: Path
    ) -> tuple[MutableMapping[str, float], Sequence[Path]]:
        """Derive additional metrics from waveform/OP artifacts that ngspice writes.

        The notebook flow emits transient/AC datasets (e.g., ``output_tran.dat``) and
        Vgs/Vth checks (``vgscheck.txt``); this method mirrors that behavior by parsing
        any ``*.dat`` waveforms for swing/THD estimates and capturing Vgs-Vth margin
        violations from the check files.
        """

        metrics: MutableMapping[str, float] = {}
        artifacts: list[Path] = []
        for candidate in workspace.glob("*.dat"):
            artifacts.append(candidate)
            waveforms = self._parse_ascii_waveform(candidate)
            metrics.update(self._derive_waveform_metrics(waveforms))
        for candidate in workspace.glob("vgscheck*.txt"):
            artifacts.append(candidate)
            metrics.update(self._parse_vgs_checks(candidate))
        return metrics, tuple(artifacts)

    def _persist_artifacts(self, artifacts: Iterable[Path], target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            try:
                destination = target_dir / artifact.name
                if artifact.resolve() == destination.resolve():
                    continue
                destination.write_bytes(artifact.read_bytes())
            except OSError:
                continue

    def _parse_ascii_waveform(self, path: Path) -> Mapping[str, Sequence[float]]:
        """Parse a whitespace-separated ASCII waveform exported by ngspice."""

        text = path.read_text().splitlines()
        rows = [line.strip() for line in text if line.strip() and not line.lstrip().startswith("*")]
        if not rows:
            return {}
        header = rows[0].split()
        columns: dict[str, list[float]] = {name: [] for name in header}
        for row in rows[1:]:
            values = row.split()
            if len(values) != len(header):
                continue
            for name, value in zip(header, values):
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    continue
        return columns

    def _derive_waveform_metrics(
        self, waveforms: Mapping[str, Sequence[float]]
    ) -> MutableMapping[str, float]:
        metrics: MutableMapping[str, float] = {}
        if not waveforms:
            return metrics

        def first_matching(keys: Sequence[str]) -> Sequence[float] | None:
            for candidate in keys:
                if candidate in waveforms:
                    return waveforms[candidate]
            for candidate in waveforms:
                if candidate.lower() in keys:
                    return waveforms[candidate]
            return None

        output_wave = first_matching(["v(out)", "out", "vout", "v(out+)", "v(o)"])
        if output_wave:
            vmax = max(output_wave)
            vmin = min(output_wave)
            metrics.setdefault("output_swing_max", vmax)
            metrics.setdefault("output_swing_min", vmin)
            metrics.setdefault("output_swing_pp", vmax - vmin)
            thd_db = self._estimate_thd(output_wave)
            if thd_db is not None:
                metrics.setdefault("thd_output_db", thd_db)

        vgs_wave = first_matching(["vgs", "vgs(node)", "vgs(m1)"])
        vth_wave = first_matching(["vth", "vth(node)", "vth(m1)"])
        if vgs_wave and vth_wave:
            margins = [vgs - vth for vgs, vth in zip(vgs_wave, vth_wave)]
            if margins:
                metrics.setdefault("vgs_vth_margin_min", min(margins))
                metrics.setdefault(
                    "vgs_vth_violations", float(sum(1 for margin in margins if margin < 0))
                )

        return metrics

    def _parse_vgs_checks(self, path: Path) -> MutableMapping[str, float]:
        """Parse Vgs/Vth check output produced by notebook control decks."""

        margins: list[float] = []
        for line in path.read_text().splitlines():
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            try:
                vgs = float(tokens[-2])
                vth = float(tokens[-1])
            except (TypeError, ValueError):
                continue
            margins.append(vgs - vth)
        metrics: MutableMapping[str, float] = {}
        if margins:
            metrics["vgs_vth_margin_min"] = min(margins)
            metrics["vgs_vth_margin_avg"] = sum(margins) / len(margins)
            metrics["vgs_vth_violations"] = float(sum(1 for margin in margins if margin < 0))
        return metrics

    def _estimate_thd(self, samples: Sequence[float]) -> float | None:
        """Lightweight THD estimator using an FFT when numpy is available."""

        try:
            import numpy as np
        except ImportError:
            return None
        if len(samples) < 4:
            return None
        arr = np.asarray(samples, dtype=float)
        arr = arr - np.mean(arr)
        spectrum = np.fft.rfft(arr)
        magnitudes = np.abs(spectrum)
        if len(magnitudes) <= 1:
            return None
        fundamental = magnitudes[1]
        harmonics = magnitudes[2:]
        if fundamental <= 0:
            return None
        harmonic_rms = float(np.sqrt(np.sum(harmonics**2)))
        if harmonic_rms <= 0:
            return None
        return 20.0 * log10(harmonic_rms / fundamental)

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
