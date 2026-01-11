from __future__ import annotations

import shutil
from pathlib import Path

from eesizer_core.contracts import SimPlan, SimRequest
from eesizer_core.contracts.enums import SimKind
from eesizer_core.runtime.context import RunContext
from eesizer_core.contracts import CircuitSource, SourceKind
from eesizer_core.sim import DeckBuildOperator, NgspiceRunOperator
from eesizer_core.metrics import ComputeMetricsOperator


def main() -> None:
    here = Path(__file__).resolve().parent
    netlist_path = here / "rc_lowpass.sp"
    circuit_source = CircuitSource(
        kind=SourceKind.spice_netlist,
        text=netlist_path.read_text(encoding="utf-8"),
        metadata={"base_dir": netlist_path.parent},
    )

    plan = SimPlan(
        sims=(
            SimRequest(
                kind=SimKind.ac,
                params={
                    "points_per_decade": 5,
                    "start_hz": 10,
                    "stop_hz": 1e6,
                    "output_nodes": ["out"],
                },
            ),
        )
    )

    deck = DeckBuildOperator().run({"circuit_source": circuit_source, "sim_plan": plan}, ctx=None).outputs["deck"]

    ctx = RunContext(workspace_root=here / "output")
    runner = NgspiceRunOperator()

    if shutil.which(runner.ngspice_bin) is None:
        print("ngspice not found; skipping run_ac_once")
        return

    result = runner.run({"deck": deck, "stage": "ac_example"}, ctx)
    raw = result.outputs["raw_data"]
    ac_csv = raw.outputs["ac_csv"]

    print(f"AC results written to {ac_csv}")
    print("First few lines:")
    for line in ac_csv.read_text(encoding="utf-8").splitlines()[:5]:
        print(line)

    metrics_op = ComputeMetricsOperator()
    metrics_result = metrics_op.run(
        {"raw_data": raw, "metric_names": ["ac_mag_db_at_1k", "ac_unity_gain_freq"]},
        ctx=None,
    )
    metrics = metrics_result.outputs["metrics"]
    print("Metrics:")
    for name, mv in metrics.values.items():
        print(f"  {name}: {mv.value} {mv.unit}")


if __name__ == "__main__":
    main()
