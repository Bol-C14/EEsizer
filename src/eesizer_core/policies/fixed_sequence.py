from __future__ import annotations

"""Policy that returns a predefined sequence of patches."""

from dataclasses import dataclass, field

from ..contracts import Patch
from ..contracts.policy import Observation, Policy


@dataclass
class FixedSequencePolicy(Policy):
    """Deterministic policy that yields a fixed list of patches then stops."""

    name: str = "fixed_sequence"
    version: str = "0.1.0"
    patches: list[Patch] = field(default_factory=list)
    _idx: int = 0

    def propose(self, obs: Observation, ctx) -> Patch:  # type: ignore[override]
        """Return the next patch in sequence or a stop signal when exhausted."""
        if self._idx < len(self.patches):
            patch = self.patches[self._idx]
            self._idx += 1
            return patch
        return Patch(stop=True, notes="sequence_exhausted")
