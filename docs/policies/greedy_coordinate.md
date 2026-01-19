# GreedyCoordinatePolicy

GreedyCoordinatePolicy is a deterministic, non-LLM baseline that behaves like a classic
coordinate descent / hill-climb optimizer. It changes one parameter at a time, adapts
step sizes based on feedback, and reacts to guard failures.

## Inputs it expects

The policy reads from `Observation.notes`:

- `current_score`: scalar objective score (lower is better).
- `best_score`: best score seen so far.
- `param_values`: `{param_id: float}` snapshot of current parameter values.
- `last_guard_report`: structured guard report from the last attempt (optional).

If `param_values` or `current_score` is missing, the policy returns `Patch(stop=True)`.

## Behavior summary

- Selects one parameter (round-robin or random).
- Tries `+step` then `-step` (mul or add depending on suffix).
- On improvement: expands step slightly and continues in the same direction.
- On failure or guard rejection: shrinks step, flips direction, retries, and
  moves on after `max_trials_per_param`.
- If a guard reason reports frozen/unknown/non-numeric params, the parameter
  is blacklisted for the run.

## Configuration knobs

Key fields (see `src/eesizer_core/policies/greedy_coordinate.py`):

- `init_step`, `min_step`, `max_step`
- `shrink`, `expand`
- `max_trials_per_param`, `max_consecutive_success`
- `selector`: `round_robin` or `random`
- `prefer_mul_suffixes`, `prefer_add_suffixes`

## Example

```python
from eesizer_core.policies import GreedyCoordinatePolicy
from eesizer_core.strategies import PatchLoopStrategy

policy = GreedyCoordinatePolicy(init_step=0.2, min_step=0.01, max_step=0.5)
strategy = PatchLoopStrategy(policy=policy)
```
