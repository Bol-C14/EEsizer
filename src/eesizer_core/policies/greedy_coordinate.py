from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..contracts import Patch, PatchOp, ParamSpace
from ..contracts.enums import PatchOpType
from ..contracts.policy import Observation, Policy


_PARAM_IN_REASON = re.compile(r"param '([^']+)'")


@dataclass
class GreedyCoordinatePolicy(Policy):
    """Coordinate descent / hill-climb policy with adaptive steps and guard feedback."""

    name: str = "greedy_coordinate"
    version: str = "0.1.0"

    init_step: float = 0.10
    min_step: float = 0.005
    max_step: float = 0.30
    shrink: float = 0.5
    expand: float = 1.05
    max_trials_per_param: int = 4
    max_consecutive_success: int = 5
    selector: str = "round_robin"
    seed: Optional[int] = None
    prefer_mul_suffixes: tuple[str, ...] = (".w", ".l", ".value", ".r", ".c")
    prefer_add_suffixes: tuple[str, ...] = (".dc",)
    add_scale_min: float = 1.0
    score_goal: str = "min"
    score_eps: float = 1e-12

    _param_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _param_index: int = field(default=0, init=False, repr=False)
    _steps: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _current_param: Optional[str] = field(default=None, init=False, repr=False)
    _phase: str = field(default="try_plus", init=False, repr=False)
    _direction: int = field(default=1, init=False, repr=False)
    _pending: bool = field(default=False, init=False, repr=False)
    _last_score: Optional[float] = field(default=None, init=False, repr=False)
    _last_param: Optional[str] = field(default=None, init=False, repr=False)
    _last_direction: int = field(default=1, init=False, repr=False)
    _last_param_values: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _trials: int = field(default=0, init=False, repr=False)
    _success_streak: int = field(default=0, init=False, repr=False)
    _blacklist: set[str] = field(default_factory=set, init=False, repr=False)
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.selector not in {"round_robin", "random"}:
            raise ValueError("selector must be 'round_robin' or 'random'")
        if self.seed is not None:
            self._rng.seed(self.seed)
        self.prefer_mul_suffixes = tuple(s.lower() for s in self.prefer_mul_suffixes)
        self.prefer_add_suffixes = tuple(s.lower() for s in self.prefer_add_suffixes)

    def propose(self, obs: Observation, ctx: Any) -> Patch:  # type: ignore[override]
        self._refresh_params(obs.param_space)

        param_values = obs.notes.get("param_values")
        if not isinstance(param_values, Mapping) or not param_values:
            return Patch(stop=True, notes="missing_param_values")
        self._last_param_values = dict(param_values)

        current_score = obs.notes.get("current_score")
        if not isinstance(current_score, (int, float)):
            return Patch(stop=True, notes="missing_current_score")

        self._apply_feedback(current_score, obs.notes)

        if (
            self._current_param is None
            or self._current_param not in self._param_ids
            or self._current_param in self._blacklist
            or not self._is_param_usable(self._current_param, param_values)
        ):
            self._advance_param(param_values)

        attempts = 0
        while attempts < max(1, len(self._param_ids)):
            param_id = self._current_param
            if param_id is None:
                break
            patch = self._make_patch(param_id, obs.param_space, param_values)
            if patch is not None:
                self._pending = True
                self._last_score = float(current_score)
                self._last_param = param_id
                self._last_direction = self._direction
                return patch
            self._advance_param(param_values)
            attempts += 1

        return Patch(stop=True, notes="no_tunable_params")

    def _refresh_params(self, param_space: ParamSpace) -> None:
        param_ids = [p.param_id for p in param_space.params if not p.frozen]
        if param_ids != self._param_ids:
            self._param_ids = param_ids
            for pid in param_ids:
                self._steps.setdefault(pid, self.init_step)
            for pid in list(self._steps.keys()):
                if pid not in self._param_ids:
                    self._steps.pop(pid, None)
            if self._param_index >= len(self._param_ids):
                self._param_index = 0

    def _apply_feedback(self, current_score: float, notes: Mapping[str, Any]) -> None:
        if not self._pending:
            return
        guard_failed, guard_params = self._guard_feedback(notes)
        if guard_params:
            for pid in guard_params:
                self._blacklist.add(pid)
        if guard_failed:
            self._register_failure()
            self._pending = False
            return
        if self._last_score is None:
            self._pending = False
            return
        if self._score_improved(current_score, self._last_score):
            self._register_success()
        else:
            self._register_failure()
        self._pending = False

    def _guard_feedback(self, notes: Mapping[str, Any]) -> tuple[bool, set[str]]:
        guard_report = notes.get("last_guard_report")
        guard_failed = False
        reasons: list[str] = []
        if isinstance(guard_report, Mapping):
            ok = guard_report.get("ok")
            if ok is False:
                guard_failed = True
                checks = guard_report.get("checks", [])
                if isinstance(checks, list):
                    for check in checks:
                        if isinstance(check, Mapping):
                            reasons.extend([str(r) for r in check.get("reasons", []) if r is not None])
        guard_failures = notes.get("last_guard_failures")
        if not guard_failed and guard_failures:
            guard_failed = True
            if isinstance(guard_failures, list):
                reasons.extend([str(r) for r in guard_failures if r is not None])

        params: set[str] = set()
        for reason in reasons:
            match = _PARAM_IN_REASON.search(reason)
            if not match:
                continue
            pid = match.group(1).lower()
            if "frozen" in reason or "unknown param" in reason or "missing current value" in reason:
                params.add(pid)
            if "non-numeric" in reason:
                params.add(pid)
        return guard_failed, params

    def _register_success(self) -> None:
        if self._last_param is None:
            return
        step = self._steps.get(self._last_param, self.init_step)
        step = min(self.max_step, step * self.expand)
        self._steps[self._last_param] = step
        self._success_streak += 1
        self._trials = 0
        self._phase = "exploit"
        if self._success_streak >= self.max_consecutive_success:
            self._advance_param(None)

    def _register_failure(self) -> None:
        if self._last_param is None:
            return
        step = self._steps.get(self._last_param, self.init_step)
        step = max(self.min_step, step * self.shrink)
        self._steps[self._last_param] = step
        self._success_streak = 0
        self._trials += 1

        if self._last_param in self._blacklist:
            self._advance_param(None)
            return

        if self._phase == "try_plus":
            self._phase = "try_minus"
            self._direction = -1
        elif self._phase == "exploit":
            self._phase = "try_minus"
            self._direction = -self._direction
        else:
            self._phase = "try_plus"
            self._direction = 1

        if self._trials >= self.max_trials_per_param:
            self._advance_param(None)

    def _advance_param(self, param_values: Mapping[str, Any] | None) -> None:
        values = param_values if param_values is not None else self._last_param_values
        self._current_param = self._select_param(values)
        self._phase = "try_plus"
        self._direction = 1
        self._trials = 0
        self._success_streak = 0

    def _select_param(self, param_values: Mapping[str, Any]) -> Optional[str]:
        if not self._param_ids:
            return None
        if self.selector == "random":
            candidates = [
                pid
                for pid in self._param_ids
                if pid not in self._blacklist and self._is_param_usable(pid, param_values)
            ]
            return self._rng.choice(candidates) if candidates else None

        for _ in range(len(self._param_ids)):
            pid = self._param_ids[self._param_index % len(self._param_ids)]
            self._param_index = (self._param_index + 1) % len(self._param_ids)
            if pid in self._blacklist:
                continue
            if not self._is_param_usable(pid, param_values):
                continue
            return pid
        return None

    @staticmethod
    def _is_param_usable(param_id: str, param_values: Mapping[str, Any]) -> bool:
        return param_id in param_values

    def _score_improved(self, new_score: float, old_score: float) -> bool:
        if self.score_goal == "max":
            return new_score > old_score + self.score_eps
        return new_score < old_score - self.score_eps

    def _op_type_for_param(self, param_id: str) -> PatchOpType:
        pid = param_id.lower()
        if any(pid.endswith(sfx) for sfx in self.prefer_add_suffixes):
            return PatchOpType.add
        if any(pid.endswith(sfx) for sfx in self.prefer_mul_suffixes):
            return PatchOpType.mul
        return PatchOpType.mul

    def _make_patch(
        self,
        param_id: str,
        param_space: ParamSpace,
        param_values: Mapping[str, Any],
    ) -> Optional[Patch]:
        if param_id in self._blacklist:
            return None
        current_val = param_values.get(param_id)
        if not isinstance(current_val, (int, float)):
            self._blacklist.add(param_id)
            return None
        param_def = param_space.get(param_id)
        step = self._steps.get(param_id, self.init_step)
        op_type = self._op_type_for_param(param_id)
        direction = self._direction

        if op_type == PatchOpType.mul:
            factor = 1.0 + direction * step
            candidate = float(current_val) * factor
            patch_op = PatchOpType.mul
            patch_value: float = factor
        else:
            scale = max(abs(float(current_val)), self.add_scale_min)
            delta = direction * step * scale
            candidate = float(current_val) + delta
            patch_op = PatchOpType.add
            patch_value = delta

        lower = param_def.lower if param_def is not None else None
        upper = param_def.upper if param_def is not None else None
        if lower is not None and candidate < lower:
            candidate = lower
            patch_op = PatchOpType.set
            patch_value = float(lower)
        if upper is not None and candidate > upper:
            candidate = upper
            patch_op = PatchOpType.set
            patch_value = float(upper)

        if abs(candidate - float(current_val)) <= self.score_eps:
            return None

        why = f"greedy_coordinate:{param_id}:{patch_op.value}:{direction:+d}:{step:.4g}"
        op = PatchOp(param=param_id, op=patch_op, value=patch_value, why=why)
        return Patch(ops=(op,))
