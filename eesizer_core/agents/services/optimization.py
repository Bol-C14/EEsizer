"""Optimization and reporting service for sizing agents."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from ...context import ArtifactKind, ExecutionContext
from ...messaging import Message, MessageRole
from ...prompts import PromptLibrary
from ..optimizer import MetricOptimizer
from ..reporting import OptimizationReporter
from ..scoring import OptimizationTargets, ScoringPolicy

logger = logging.getLogger(__name__)


class OptimizationService:
    """Wraps the optimization loop plus reporting artifacts."""

    def __init__(
        self,
        *,
        optimizer: MetricOptimizer,
        scoring: ScoringPolicy,
        prompts: PromptLibrary,
        targets: OptimizationTargets,
        goal: str,
        tolerance_percent: float,
    ) -> None:
        self.optimizer = optimizer
        self.scoring = scoring
        self.prompts = prompts
        self.targets = targets
        self.goal = goal
        self.tolerance_percent = tolerance_percent

    def optimize(self, context: ExecutionContext, metrics: Mapping[str, float]) -> MutableMapping[str, float]:
        """Run optimization and emit reports/artifacts."""

        baseline_metrics = dict(metrics)
        logger.info(f"Starting optimization loop. Baseline metrics: {baseline_metrics}")
        result = self.optimizer.optimize(context, metrics)
        optimized = dict(result.metrics)
        logger.info(
            "Optimization finished. Best score: %s, Iterations: %s",
            result.best_score,
            len(result.history),
        )

        optimized["iterations"] = float(len(result.history))
        optimized["meets_gain"] = float(optimized.get("gain_db", 0.0) >= self.targets.gain_db)
        optimized["meets_power"] = float(optimized.get("power_mw", 0.0) <= self.targets.power_mw)
        optimized["targets_met"] = float(self.scoring.meets_targets(optimized))
        optimized["composite_score"] = float(result.best_score)

        self._emit_target_summary(context)
        artifacts_dir = self._resolve_dir(context, "artifacts")
        reporter = OptimizationReporter(artifacts_dir)
        reporter.write_summary(context, optimized)
        reporter.write_history(context, result.history)
        reporter.write_variant_comparison(
            context,
            variants=(
                ("baseline", baseline_metrics),
                ("optimized", optimized),
            ),
            scoring_fn=self.scoring.score,
        )

        if context.netlist_path:
            copied_netlist = artifacts_dir / context.netlist_path.name
            if not copied_netlist.exists():
                shutil.copy2(context.netlist_path, copied_netlist)
            context.attach_artifact(
                "netlist_copy",
                copied_netlist,
                kind=ArtifactKind.NETLIST,
                description="Input netlist snapshot captured for reproducibility.",
            )

        return optimized

    # Internal helpers

    def _emit_target_summary(self, context: ExecutionContext) -> None:
        template = self.prompts.load("target_value_system_prompt")
        content = template.render(
            goal=self.goal,
            target_gain_db=self.targets.gain_db,
            target_power_mw=self.targets.power_mw,
            tolerance_percent=self.tolerance_percent,
        )
        context.log(
            Message(role=MessageRole.SYSTEM, content=content, name="spec_checker")
        )

    def _resolve_dir(self, context: ExecutionContext, category: str) -> Path:
        if context.paths:
            mapping = {
                "simulations": context.paths.simulations,
                "artifacts": context.paths.artifacts,
                "logs": context.paths.logs,
                "plans": context.paths.plans,
            }
            target = mapping.get(category)
            if target:
                target.mkdir(parents=True, exist_ok=True)
                return target
        context.working_dir.mkdir(parents=True, exist_ok=True)
        return context.working_dir


__all__ = ["OptimizationService"]
