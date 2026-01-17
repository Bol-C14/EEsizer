import json
from pathlib import Path

from eesizer_core.baselines.legacy_metrics_adapter import LegacyMetricsBaseline, LegacyModules
from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective
from eesizer_core.contracts.enums import SourceKind, StopReason
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.context import RunContext


class FakeLegacySimUtils:
    __file__ = "fake_legacy_sim_utils.py"

    def ac_simulation(self, netlist, input_name, output_node, output_dir):
        return netlist

    def dc_simulation(self, netlist, input_name, output_node, output_dir):
        return netlist

    def trans_simulation(self, netlist, input_name, output_node, output_dir):
        return netlist

    def run_ngspice(self, circuit, filename, output_dir: str, timeout_s: int = 120):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return True

    def ac_gain(self, file_name, output_dir: str):
        return -12.3

    def bandwidth(self, file_name, output_dir: str):
        return 1e6

    def unity_bandwidth(self, file_name, output_dir: str):
        return 2e6


def test_legacy_baseline_writes_run_outputs(tmp_path):
    netlist = "M1 d g s b nmos W=1u L=0.1u\n.end\n"
    source = CircuitSource(kind=SourceKind.spice_netlist, text=netlist)
    spec = CircuitSpec(
        objectives=(
            Objective(metric="gain_db", target=-20.0, sense="ge"),
            Objective(metric="ugbw_hz", target=1e6, sense="ge"),
        )
    )

    legacy = LegacyModules(simulation_utils=FakeLegacySimUtils())
    strategy = LegacyMetricsBaseline(legacy_modules=legacy)
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=1, no_improve_patience=1))
    ctx = RunContext(workspace_root=tmp_path)

    result = strategy.run(spec=spec, source=source, ctx=ctx, cfg=cfg)
    run_dir = ctx.run_dir()

    assert result.stop_reason == StopReason.baseline_legacy
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "inputs" / "source.sp").exists()
    assert (run_dir / "history" / "iterations.jsonl").exists()
    assert (run_dir / "provenance" / "operator_calls.jsonl").exists()
    assert (run_dir / "best" / "best.sp").exists()
    assert (run_dir / "best" / "best_metrics.json").exists()

    lines = (run_dir / "history" / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["iteration"] == 0

    metrics = json.loads((run_dir / "best" / "best_metrics.json").read_text(encoding="utf-8"))
    assert "gain_db" in metrics
    assert metrics["gain_db"]["value"] == -12.3
