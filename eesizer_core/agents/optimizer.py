"""Reusable optimization loop for sizing agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from ..context import ExecutionContext
from ..messaging import Message, MessageRole
from ..prompts import PromptLibrary
from .scoring import OptimizationTargets, ScoringPolicy


@dataclass(slots=True)
class OptimizationResult:
    """Summary returned by the optimizer."""

    metrics: MutableMapping[str, float]
    history: Sequence[MutableMapping[str, float]]
    best_score: float


class MetricOptimizer:
    """Encapsulates the iterative optimization loop."""

    def __init__(
        self,
        *,
        scoring: ScoringPolicy,
        prompts: PromptLibrary,
        targets: OptimizationTargets,
        max_iterations: int,
        nudge_fn: Callable[[Mapping[str, float]], MutableMapping[str, float]],
        stagnation_rounds: int = 3,
        min_improvement: float = 0.01,
        pass_keys: Iterable[str] = ("pass_", "pass", "all_pass"),
    ) -> None:
        self.scoring = scoring
        self.prompts = prompts
        self.targets = targets
        self.max_iterations = max_iterations
        self.nudge_fn = nudge_fn
        self.stagnation_rounds = max(1, stagnation_rounds)
        self.min_improvement = max(0.0, min_improvement)
        self.pass_keys = tuple(pass_keys)

    def optimize(
        self,
        context: ExecutionContext,
        metrics: Mapping[str, float],
    ) -> OptimizationResult:
        """Run a scoring-guided loop that logs prompts and returns the best metrics."""

        optimized = dict(metrics)
        best_metrics = dict(optimized)
        best_score = self.scoring.score(optimized)
        tolerance = self.scoring.tolerance
        stagnation_counter = 0

        strategy_note = self.prompts.load("optimization_strategy").render(
            target_gain_db=self.targets.gain_db,
            target_power_mw=self.targets.power_mw,
            tolerance_percent=tolerance * 100,
        )
        context.log(
            Message(
                role=MessageRole.SYSTEM,
                content=strategy_note,
                name="optimization_strategy",
            )
        )

        history: list[MutableMapping[str, float]] = []
        if self._has_pass_flag(optimized):
            return OptimizationResult(metrics=best_metrics, history=tuple(history), best_score=best_score)

        for iteration in range(1, self.max_iterations + 1):
            analysis_note = self.prompts.load("analysing_system_prompt").render(
                iteration=iteration,
                gain_db=optimized.get("gain_db", 0.0),
                power_mw=optimized.get("power_mw", 0.0),
                bandwidth_hz=optimized.get("bandwidth_hz", 0.0),
                transistor_count=optimized.get("transistor_count", 0.0),
            )
            context.log(
                Message(role=MessageRole.USER, content=analysis_note, name="analysis_prompt")
            )

            optimization_note = self.prompts.load("optimization_prompt").render(
                iteration=iteration,
                power_budget=self.targets.power_mw,
                target_gain_db=self.targets.gain_db,
            )
            context.log(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=optimization_note,
                    name="optimization_prompt_logger",
                )
            )

            next_metrics = self.nudge_fn(optimized)
            gain_delta = next_metrics.get("gain_db", 0.0) - optimized.get("gain_db", 0.0)
            power_delta = next_metrics.get("power_mw", 0.0) - optimized.get("power_mw", 0.0)
            sizing_note = self.prompts.load("sizing_prompt").render(
                iteration=iteration,
                gain_delta=gain_delta,
                power_delta=power_delta,
            )
            context.log(
                Message(role=MessageRole.ASSISTANT, content=sizing_note, name="sizing_logger")
            )

            current_score = self.scoring.score(optimized)
            next_score = self.scoring.score(next_metrics)
            if self.scoring.should_accept(current_score, next_score):
                optimized = next_metrics
            if next_score > best_score:
                best_score = next_score
                best_metrics = dict(next_metrics)
                stagnation_counter = 0
            else:
                improvement = (
                    (next_score - best_score) / max(best_score, 1e-9)
                    if best_score > 0
                    else next_score
                )
                stagnation_counter = stagnation_counter + 1 if improvement < self.min_improvement else 0

            history.append(
                {
                    "iteration": float(iteration),
                    "gain_db": float(optimized.get("gain_db", 0.0)),
                    "power_mw": float(optimized.get("power_mw", 0.0)),
                    "analysis_note": analysis_note,
                    "optimization_note": optimization_note,
                    "sizing_note": sizing_note,
                }
            )

            if self._has_pass_flag(next_metrics) or self.scoring.meets_targets(optimized):
                break
            if stagnation_counter >= self.stagnation_rounds:
                break

        return OptimizationResult(metrics=best_metrics, history=tuple(history), best_score=best_score)

    def _has_pass_flag(self, metrics: Mapping[str, float]) -> bool:
        for key in self.pass_keys:
            try:
                if bool(metrics.get(key)):  # type: ignore[arg-type]
                    return True
            except Exception:
                continue
        return False


__all__ = ["MetricOptimizer", "OptimizationResult"]
