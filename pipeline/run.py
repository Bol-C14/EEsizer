"""Command-line entry point that mirrors the notebook pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from eesizer_core import (
    AgentConfig,
    ConfigLoader,
    ContextManager,
    MockNgSpiceSimulator,
    OptimizationTargets,
    SimpleSizingAgent,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_model": "gpt-4o-mini",
    "simulation": {"binary_path": "ngspice", "timeout_seconds": 120, "working_root": "output/pipeline"},
    "optimization": {"max_iterations": 10, "tolerance_percent": 0.05, "vgs_margin_volts": 0.05},
    "agents": {
        "simple": {
            "model": "gpt-4o-mini",
        }
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the notebook pipeline without Jupyter")
    parser.add_argument("--netlist", required=True, type=Path, help="Path to the input SPICE netlist")
    parser.add_argument("--goal", required=True, help="Design goal that guides planning")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file")
    parser.add_argument("--target-gain", type=float, default=55.0, help="Gain target in dB")
    parser.add_argument("--target-power", type=float, default=5.0, help="Max power consumption in mW")
    parser.add_argument("--workdir", type=Path, help="Override working directory root")
    parser.add_argument("--run-id", help="Custom run identifier (defaults to a UUID4)")
    return parser.parse_args(argv)


def load_config(path: Path | None) -> Dict[str, Any]:
    if not path:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyyaml is required to read YAML configs") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format: {suffix}")


def build_agent_config(data: Dict[str, Any]) -> AgentConfig:
    loader = ConfigLoader(data)
    agents = data.get("agents", {})
    raw_agent = agents.get("simple", agents.get("default", {}))
    return loader.build_agent("simple", raw_agent)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_data = load_config(args.config)
    if args.workdir:
        config_data.setdefault("simulation", {})["working_root"] = str(args.workdir)
    agent_config = build_agent_config(config_data)
    sim_config = agent_config.simulation
    simulator = MockNgSpiceSimulator(sim_config)
    targets = OptimizationTargets(gain_db=args.target_gain, power_mw=args.target_power)
    agent = SimpleSizingAgent(agent_config, simulator, goal=args.goal, targets=targets)

    run_id = args.run_id or uuid.uuid4().hex[:8]
    working_root = Path(agent_config.simulation.working_root)
    working_root.mkdir(parents=True, exist_ok=True)

    with ContextManager(run_id=run_id, base_dir=working_root, config_name="cli") as ctx:
        ctx.netlist_path = args.netlist
        ctx.metadata["goal"] = args.goal
        result = agent.run(ctx)

    summary = {
        "run_id": run_id,
        "artifacts": result.artifacts,
        "metrics": result.metrics,
    }
    print(json.dumps(summary, indent=2))
    result_path = working_root / run_id / "pipeline_result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
