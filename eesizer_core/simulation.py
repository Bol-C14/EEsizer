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
from .spice import (
    ControlDeck,
    WaveformExport,
    augment_netlist,
    load_measurements,
    parse_measure_log,
)


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

        command = [str(self.config.binary_path), "-b", "-o", log_path.name, netlist_path.name]
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
            wave_metrics, artifacts = self._collect_waveform_metrics(workspace, deck)
            metrics.update(wave_metrics)
            if artifact_dir:
                self._persist_artifacts(artifacts, artifact_dir)
            return metrics
        except subprocess.CalledProcessError as exc:
            print(f"ngspice failed with exit code {exc.returncode}")
            print(f"STDOUT:\n{exc.stdout}")
            print(f"STDERR:\n{exc.stderr}")
            if log_path.exists():
                print(f"LOG:\n{log_path.read_text()}")
            raise RuntimeError(f"ngspice binary failed: {self.config.binary_path}") from exc
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
        self, workspace: Path, deck: ControlDeck | None = None
    ) -> tuple[MutableMapping[str, float], Sequence[Path]]:
        """Derive metrics from waveform dumps plus any Vgs/Vth check files."""

        metrics: MutableMapping[str, float] = {}
        artifacts: list[Path] = []
        waveforms: MutableMapping[str, Sequence[float]] = {}
        exports = tuple(deck.waveform_exports) if isinstance(deck, ControlDeck) else ()
        processed_dat: set[str] = set()
        for export in exports:
            candidate = workspace / export.filename
            if not candidate.exists():
                continue
            artifacts.append(candidate)
            processed_dat.add(candidate.name)
            parsed = self._parse_wrdata_file(candidate, export.vectors)
            for name, values in parsed.items():
                if not values:
                    continue
                waveforms.setdefault(name.lower(), values)
        if not waveforms:
            for candidate in workspace.glob("*.dat"):
                if candidate.name in processed_dat:
                    continue
                artifacts.append(candidate)
                parsed = self._parse_ascii_waveform(candidate)
                for name, values in parsed.items():
                    if not values or name in waveforms:
                        continue
                    waveforms[name] = values
        metrics.update(self._derive_waveform_metrics(waveforms))
        for candidate in workspace.glob("vgscheck*.txt"):
            artifacts.append(candidate)
            metrics.update(self._parse_vgs_checks(candidate))
        return metrics, tuple(artifacts)

    def _parse_wrdata_file(
        self, path: Path, vectors: Sequence[str]
    ) -> Mapping[str, Sequence[float]]:
        columns: dict[str, list[float]] = {vector: [] for vector in vectors}
        expected = len(vectors) * 2
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return columns
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < expected:
                continue
            try:
                real_values = [float(parts[idx]) for idx in range(0, expected, 2)]
            except (TypeError, ValueError):
                continue
            for vector, value in zip(vectors, real_values):
                columns[vector].append(value)
        return columns

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
        columns: dict[str, list[float]] = {name.lower(): [] for name in header}
        for row in rows[1:]:
            values = row.split()
            if len(values) != len(header):
                continue
            for name, value in zip(header, values):
                try:
                    columns[name.lower()].append(float(value))
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
            lowered = tuple(key.lower() for key in keys)
            for candidate in lowered:
                if candidate in waveforms:
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

        supply_current_key = next((key for key in waveforms if key.startswith("i(")), None)
        supply_voltage_wave: Sequence[float] | None = None
        supply_current_wave: Sequence[float] | None = None
        if supply_current_key:
            supply_current_wave = waveforms[supply_current_key]
            branch = supply_current_key[2:-1]
            candidate_keys = [f"v({branch})", branch]
            for candidate in candidate_keys:
                key = candidate.lower()
                if key in waveforms:
                    supply_voltage_wave = waveforms[key]
                    break
        if (
            supply_voltage_wave
            and supply_current_wave
            and len(supply_voltage_wave) == len(supply_current_wave)
            and supply_voltage_wave
        ):
            samples = [-v * i * 1e3 for v, i in zip(supply_voltage_wave, supply_current_wave)]
            if samples:
                metrics.setdefault("power_mw", float(sum(samples) / len(samples)))

        # AC Analysis Metrics
        freq_wave = waveforms.get("frequency")
        if freq_wave:
            vdb_key = next((k for k in waveforms if k.startswith("vdb(")), None)
            if vdb_key:
                vdb_wave = waveforms[vdb_key]
                if len(vdb_wave) == len(freq_wave) and vdb_wave:
                    max_gain = max(vdb_wave)
                    metrics.setdefault("ac_gain_db", max_gain)
                    
                    # Bandwidth (-3dB)
                    target_3db = max_gain - 3.0
                    for f, g in zip(freq_wave, vdb_wave):
                        if g <= target_3db:
                            metrics.setdefault("bandwidth_hz", f)
                            break
                    
                    # Unity Bandwidth (0dB)
                    for f, g in zip(freq_wave, vdb_wave):
                        if g <= 0.0:
                            metrics.setdefault("unity_bandwidth_hz", f)
                            break

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
