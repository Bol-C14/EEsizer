# LLMPatchPolicy

LLMPatchPolicy formats prompts and parses Patch JSON while keeping policy logic pure.
It never emits a full netlist and only returns patch operations.
LLM calls and retries are handled by the strategy via `LLMCallOperator` for provenance and audit artifacts.
The policy exposes `build_request()` and `parse_response()` helpers for the strategy to use.

## Inputs it expects

The policy reads from `Observation.notes`:

- `current_score`: numeric score for the current candidate.
- `best_score`: best score so far.
- `param_values`: `{param_id: value}` snapshot of current parameter values.
- `last_guard_report` or `last_guard_failures`: optional guard feedback for rejection/blacklist hints.
- `attempt`: retry index inside the current iteration (added by PatchLoopStrategy).

If `param_values` or `current_score` is missing, the policy returns a stop reason to the strategy.

The prompt clarifies that score is a penalty to minimize and that `score=0.0` means all objectives are satisfied.

## Output contract

The LLM response must conform to `docs/templates/patch.schema.json`:

```json
{
  "patch": [
    {"param": "r1.value", "op": "mul", "value": 0.9, "why": "reduce attenuation"}
  ],
  "stop": false,
  "notes": "optional"
}
```

Unknown fields are rejected. Parameters must be in the non-frozen `ParamSpace`.

## Retry behavior

If the response does not parse:

1. The strategy retries up to `max_retries` times.
2. Each retry appends the parse error to the prompt and re-requests JSON only.
3. If still failing, it returns `Patch(stop=True, notes="llm_parse_failed")`.

## Run artifacts

Each LLM call is recorded under:

```
runs/<run_id>/llm/llm_i{iter}_a{attempt}/
  request.json
  prompt.txt
  response.txt
  parsed_patch.json
  parse_error.txt
  call_error.txt
```

Retries append `_rXX` to the stage directory name.

## Mock backend behavior

When `provider="mock"` and no explicit mock response is supplied, the policy generates
single-op patches with adaptive step sizes based on the current score and clamps
values to any defined bounds (with a conservative fallback clamp when bounds are absent).
