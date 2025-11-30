"""Command-line entry point that mirrors the notebook pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict

from eesizer_core import (
    AgentConfig,
    Claude35Agent,
    ConfigLoader,
    ContextManager,
    Gemini30Agent,
    Gpt4oAgent,
    Gpt5Agent,
    Gpt5MiniAgent,
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
        "openai": {
            "kind": "openai",
            "credentials": {"api_key": ""},
            "parameters": {"base_url": None},
            "description": "Default OpenAI provider wiring for GPT agents.",
        },
        "anthropic": {
            "kind": "anthropic",
            "credentials": {"api_key": ""},
            "parameters": {"base_url": None},
            "description": "Default Anthropic provider wiring for Claude agents.",
        },
        "gemini": {
            "kind": "gemini",
            "credentials": {"api_key": ""},
            "parameters": {},
            "description": "Default Gemini provider wiring for Google agents.",
        },
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
            "tools": ["ngspice", "mock-ngspice"],
            "description": "Reference agent wiring together planning, simulation, and optimization.",
        },
        "gpt5": {
            "model": "gpt-5.1",
            "tools": ["openai", "ngspice", "mock-ngspice"],
            "description": "Notebook-aligned GPT-5 agent using OpenAI provider and ngspice backend.",
        },
        "gpt4o": {
            "model": "gpt-4o",
            "tools": ["openai", "ngspice", "mock-ngspice"],
            "description": "Fast-iteration GPT-4o agent wired for ngspice simulations.",
        },
        "gpt5mini": {
            "model": "gpt-5.1-mini",
            "tools": ["openai", "ngspice", "mock-ngspice"],
            "description": "Cost-efficient GPT-5.1 mini agent mirroring the notebook flow.",
        },
        "claude35": {
            "model": "claude-3.5-sonnet",
            "tools": ["anthropic", "ngspice", "mock-ngspice"],
            "description": "Claude 3.5 agent with Anthropic provider and ngspice simulation.",
        },
        "gemini30": {
            "model": "gemini-3.0-pro",
            "tools": ["gemini", "ngspice", "mock-ngspice"],
            "description": "Gemini 3.0 agent with Google provider and ngspice simulation.",
        }
    },
}

AGENT_REGISTRY = {
    "simple": SimpleSizingAgent,
    "gpt5": Gpt5Agent,
    "gpt4o": Gpt4oAgent,
    "gpt5mini": Gpt5MiniAgent,
    "claude35": Claude35Agent,
    "gemini30": Gemini30Agent,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the notebook pipeline without Jupyter")
    parser.add_argument("--netlist", required=True, type=Path, help="Path to the input SPICE netlist")
    parser.add_argument("--goal", required=True, help="Design goal that guides planning")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file")
    parser.add_argument(
        "--agent",
        default="simple",
        help=(
            "Agent name from the orchestrator config to run (e.g. simple, gpt5, gpt4o, gpt5mini,"
            " claude35, gemini30)."
        ),
    )
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
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Force live LLM calls when credentials are configured (defaults to recorded fixtures).",
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


def select_agent_config(config: OrchestratorConfig, preferred: str | None = None) -> tuple[str, AgentConfig]:
    if preferred and preferred in config.agents:
        return preferred, config.agents[preferred]
    if config.agents:
        # Deterministic iteration order for Python 3.8+ dictionaries.
        name, cfg = next(iter(config.agents.items()))
        return name, cfg
    raise ValueError("No agents available in the orchestrator configuration")


def build_simulator(agent_config: AgentConfig, orchestrator_config: OrchestratorConfig):
    allow_real = os.environ.get("EESIZER_ENABLE_REAL_NGSPICE", "").lower() in {"1", "true", "yes"}
    for tool_name in agent_config.tools:
        tool_cfg = orchestrator_config.tools.get(tool_name)
        if not tool_cfg:
            continue
        if tool_cfg.kind == "ngspice":
            if not allow_real:
                continue
            binary_value = tool_cfg.parameters.get("binary") or agent_config.simulation.binary_path
            binary_path = Path(binary_value)
            if binary_path.exists() or shutil.which(str(binary_path)):
                agent_config.simulation.binary_path = binary_path
                include_paths = [Path(p) for p in tool_cfg.parameters.get("include_paths", [])]
                return NgSpiceRunner(agent_config.simulation, include_paths=include_paths)
            continue
        if tool_cfg.kind in {"simulator", "mock"}:
            break
    return MockNgSpiceSimulator(agent_config.simulation)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_data = load_config(args.config)
    if args.workdir:
        config_data.setdefault("output_paths", {})["root"] = str(args.workdir)
    orchestrator_config = build_orchestrator_config(config_data)
    agent_name, agent_config = select_agent_config(orchestrator_config, preferred=args.agent)
    simulator = build_simulator(agent_config, orchestrator_config)
    targets = OptimizationTargets(gain_db=args.target_gain, power_mw=args.target_power)
    agent_cls = AGENT_REGISTRY.get(agent_name)
    if not agent_cls:
        known = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Unknown agent '{agent_name}'. Available agents: {known}")
    agent = agent_cls(
        agent_config,
        simulator,
        goal=args.goal,
        targets=targets,
        tool_configs=orchestrator_config.tools,
        force_live_llm=args.live_llm,
    )

    run_id = args.run_id or uuid.uuid4().hex[:8]
    output_policy = agent_config.resolve_output_paths(orchestrator_config.output_paths)
    layout = output_policy.build_layout(run_id, netlist_stem=args.netlist.stem)
    run_dir = layout.run_dir
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
        path_layout=layout,
    ) as ctx:
        ctx.netlist_path = args.netlist
        ctx.metadata["goal"] = args.goal
        ctx.metadata["agent"] = {"name": agent_config.name, "model": agent_config.model}
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
        "agent": {"name": agent_config.name, "model": agent_config.model},
    }
    print(json.dumps(summary, indent=2))
    result_path = run_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
