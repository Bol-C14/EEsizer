import json
import os
from pathlib import Path

import json
import os
from pathlib import Path

import pytest

from eesizer_core.simulation import MockNgSpiceSimulator


class RichMockNgSpiceSimulator(MockNgSpiceSimulator):
    """Return an expanded measurement payload covering the notebook suite."""

    def run(self, netlist_text: str, deck, workdir: Path | None = None, artifact_dir=None):  # type: ignore[override]
        base = super().run(netlist_text, deck, workdir, artifact_dir=artifact_dir)
        base.update(
            {
                "ac_gain_db": base["gain_db"],
                "tran_gain_db": base["gain_db"] - 0.5,
                "output_swing_max": 1.2,
                "output_swing_min": 0.2,
                "offset_v": 0.01,
                "icmr_min_v": 0.05,
                "icmr_max_v": 1.5,
                "thd_output_db": -42.0,
                "cmrr_db": 72.0,
                "bandwidth_hz": 2.5e6,
                "unity_bandwidth_hz": 5.0e6,
            }
        )
        return base


@pytest.fixture
def mini_netlist(tmp_path: Path) -> Path:
    fixture_path = Path(__file__).parent / "fixtures" / "mini_ota.cir"
    target = tmp_path / "mini_ota.cir"
    target.write_text(fixture_path.read_text())
    return target


@pytest.fixture
def recorded_llm_responses() -> dict:
    path = Path(__file__).parent / "fixtures" / "llm_responses.json"
    return json.loads(path.read_text())


@pytest.fixture
def recorded_only_env(monkeypatch):
    monkeypatch.delenv("EESIZER_LIVE_LLM", raising=False)
    return os.getenv("EESIZER_LIVE_LLM")


@pytest.fixture
def rich_mock_simulator() -> RichMockNgSpiceSimulator:
    from eesizer_core.config import SimulationConfig

    return RichMockNgSpiceSimulator(SimulationConfig(binary_path=Path("ngspice")))
