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
    NgSpiceRunner,
    OptimizationTargets,
    OrchestratorConfig,
    SimpleSizingAgent,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_model": "gpt-4o-mini",
    "simulation": {"binary_path": "ngspice", "timeout_seconds": 120},
    "optimization": {"max_iterations": 10, "tolerance_percent": 0.05, "vgs_margin_volts": 0.05},
    "output_paths": {
        "root": "output/pipeline",
        "artifacts_dir": "artifacts",
        "logs_dir": "logs",
        "simulations_dir": "simulations",
        "plans_dir": "plans",
    },
    "tools": {
        "mock-ngspice": {
            "kind": "simulator",
            "parameters": {"binary": "ngspice"},
            "description": "Deterministic stub used for tests until real ngspice integration lands.",
        },
        "ngspice": {
            "kind": "ngspice",
            "parameters": {
                "binary": "ngspice",
                "include_paths": ["agent_test_gpt", "agent_test_gemini"],
            },
            "description": "Real ngspice backend mirroring the notebook execution flow.",
        }
    },
    "agents": {
        "simple": {
            "model": "gpt-4o-mini",
            "tools": ["mock-ngspice"],
            "description": "Reference agent wiring together planning, mock simulation, and optimization.",
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
    parser.add_argument("--corner", default="tt", help="Process corner descriptor (e.g. tt, ff, ss)")
    parser.add_argument("--supply-voltage", type=float, default=1.8, help="Supply voltage in volts")
    parser.add_argument(
        "--temperature",
        type=float,
        default=27.0,
        help="Ambient temperature in Celsius",
    )
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


def build_orchestrator_config(data: Dict[str, Any]) -> OrchestratorConfig:
    loader = ConfigLoader(data)
    return loader.build_orchestrator()


def select_agent_config(config: OrchestratorConfig, preferred: str | None = None) -> AgentConfig:
    if preferred and preferred in config.agents:
        return config.agents[preferred]
    if config.agents:
        # Deterministic iteration order for Python 3.8+ dictionaries.
        return next(iter(config.agents.values()))
    raise ValueError("No agents available in the orchestrator configuration")


def build_simulator(agent_config: AgentConfig, orchestrator_config: OrchestratorConfig):
    for tool_name in agent_config.tools:
        tool_cfg = orchestrator_config.tools.get(tool_name)
        if not tool_cfg:
            continue
        if tool_cfg.kind == "ngspice":
            include_paths = [Path(p) for p in tool_cfg.parameters.get("include_paths", [])]
            return NgSpiceRunner(agent_config.simulation, include_paths=include_paths)
        if tool_cfg.kind in {"simulator", "mock"}:
            break
    return MockNgSpiceSimulator(agent_config.simulation)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_data = load_config(args.config)
    if args.workdir:
        config_data.setdefault("output_paths", {})["root"] = str(args.workdir)
    orchestrator_config = build_orchestrator_config(config_data)
    agent_config = select_agent_config(orchestrator_config, preferred="simple")
    simulator = build_simulator(agent_config, orchestrator_config)
    targets = OptimizationTargets(gain_db=args.target_gain, power_mw=args.target_power)
    agent = SimpleSizingAgent(agent_config, simulator, goal=args.goal, targets=targets)

    run_id = args.run_id or uuid.uuid4().hex[:8]
    output_policy = agent_config.resolve_output_paths(orchestrator_config.output_paths)
    run_dir = output_policy.ensure_run_structure(run_id)
    working_root = output_policy.root

    def _pre_run(ctx):
        ctx.metadata["lifecycle_state"] = "starting"

    def _post_run(ctx):
        ctx.metadata["lifecycle_state"] = "finished"

    with ContextManager(
        run_id=run_id,
        base_dir=working_root,
        config_name="cli",
        pre_run_hook=_pre_run,
        post_run_hook=_post_run,
    ) as ctx:
        ctx.netlist_path = args.netlist
        ctx.metadata["goal"] = args.goal
        ctx.set_environment(
            corner=args.corner,
            supply_voltage=args.supply_voltage,
            temperature_c=args.temperature,
        )
        result = agent.run(ctx)

    summary = {
        "run_id": run_id,
        "artifacts": result.serialize_artifacts(),
        "metrics": result.metrics,
        "environment": ctx.environment.to_dict(),
        "sim_output": ctx.metadata.get("sim_output"),
    }
    print(json.dumps(summary, indent=2))
    result_path = run_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
