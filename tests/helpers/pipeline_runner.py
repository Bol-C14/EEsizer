"""Test helper for invoking the pipeline end-to-end."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
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
from pipeline import run as pipeline_run


@dataclass(slots=True)
class PipelineRunResult:
    """Structured result returned by :func:`run_pipeline_for_test`."""

    summary: Dict[str, Any]
    run_dir: Path
    backend: str


def has_real_ngspice() -> bool:
    """Return ``True`` if a usable ngspice binary is available."""

    if os.environ.get("EESIZER_ENABLE_REAL_NGSPICE") == "0":
        return False
    binary = shutil.which("ngspice")
    if not binary:
        return False
    env_hint = os.environ.get("EESIZER_ENABLE_REAL_NGSPICE")
    return bool(binary and (env_hint is None or env_hint.lower() in {"1", "true", "yes"}))


def _build_orchestrator(use_real_ngspice: bool, workdir: Path | None) -> OrchestratorConfig:
    config_data = json.loads(json.dumps(pipeline_run.DEFAULT_CONFIG))
    agent_entry = config_data.setdefault("agents", {}).setdefault("simple", {})
    agent_entry["tools"] = ["ngspice" if use_real_ngspice else "mock-ngspice"]
    if workdir:
        config_data.setdefault("output_paths", {})["root"] = str(workdir)
    loader = ConfigLoader(config_data)
    return loader.build_orchestrator()


def _select_agent(config: OrchestratorConfig) -> AgentConfig:
    if config.agents:
        return next(iter(config.agents.values()))
    raise ValueError("No agents available in orchestrator configuration")


def _build_simulator(agent_config: AgentConfig, orchestrator_config: OrchestratorConfig):
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


def run_pipeline_for_test(
    *,
    netlist: Path,
    goal: str,
    targets: OptimizationTargets,
    run_id: str | None = None,
    workdir: Path | None = None,
    use_real_ngspice: bool = False,
) -> PipelineRunResult:
    """Run the pipeline in-process and return the parsed summary."""

    orchestrator_config = _build_orchestrator(use_real_ngspice, workdir)
    agent_config = _select_agent(orchestrator_config)
    simulator = _build_simulator(agent_config, orchestrator_config)
    agent = SimpleSizingAgent(agent_config, simulator, goal=goal, targets=targets)

    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    output_policy = agent_config.resolve_output_paths(orchestrator_config.output_paths)
    run_dir = output_policy.ensure_run_structure(resolved_run_id)
    working_root = output_policy.root

    def _pre_run(ctx):
        ctx.metadata["lifecycle_state"] = "starting"

    def _post_run(ctx):
        ctx.metadata["lifecycle_state"] = "finished"

    with ContextManager(
        run_id=resolved_run_id,
        base_dir=working_root,
        config_name="test",
        pre_run_hook=_pre_run,
        post_run_hook=_post_run,
    ) as ctx:
        ctx.netlist_path = netlist
        ctx.metadata["goal"] = goal
        ctx.set_environment()
        result = agent.run(ctx)

    summary = {
        "run_id": resolved_run_id,
        "artifacts": result.serialize_artifacts(),
        "metrics": result.metrics,
        "environment": ctx.environment.to_dict(),
        "sim_output": ctx.metadata.get("sim_output"),
    }
    result_path = run_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(summary, indent=2))

    backend = "ngspice" if use_real_ngspice else "mock"
    return PipelineRunResult(summary=summary, run_dir=run_dir, backend=backend)


__all__ = ["PipelineRunResult", "has_real_ngspice", "run_pipeline_for_test"]
