from __future__ import annotations

"""Randomized policy that nudges one parameter per iteration."""

import random
from dataclasses import dataclass, field
from typing import Optional

from ..contracts import Patch, PatchOp, ParamSpace
from ..contracts.enums import PatchOpType
from ..contracts.policy import Observation, Policy


@dataclass
class RandomNudgePolicy(Policy):
    """Simple heuristic policy: nudge one random non-frozen parameter.

    The policy proposes a multiplicative update (PatchOpType.mul) by a small
    relative step in either direction.
    """

    name: str = "random_nudge"
    version: str = "0.1.0"
    step: float = 0.02
    seed: Optional[int] = None
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        """Seed the internal RNG when a fixed seed is supplied."""
        if self.seed is not None:
            self._rng.seed(self.seed)

    def _pick_param(self, param_space: ParamSpace) -> Optional[str]:
        """Return a random tunable parameter id, or None if none are available."""
        candidates = [p.param_id for p in param_space.params if not p.frozen]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def propose(self, obs: Observation, ctx) -> Patch:  # type: ignore[override]
        """Propose a single-step multiplicative patch or a stop signal."""
        param_id = self._pick_param(obs.param_space)
        if param_id is None:
            return Patch(stop=True, notes="no_tunable_params")

        direction = self._rng.choice([-1.0, 1.0])
        factor = 1.0 + direction * self.step
        # Multiplicative nudges keep the current value as the implicit baseline.
        op = PatchOp(param=param_id, op=PatchOpType.mul, value=factor, why="random_nudge")
        return Patch(ops=(op,))
