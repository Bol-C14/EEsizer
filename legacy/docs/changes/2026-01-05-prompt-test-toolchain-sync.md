# Change Note: Prompt / Test / Toolchain Sync (2026-01-05)

Summary
- Synchronized prompt wording, test expectations, and toolchain behavior to reflect the policy that `simulation_*` functions run NGspice internally and the LLM should only emit JSON `tool_calls` listing simulation/analysis names.

What changed
- `agent_test_gpt/prompts.py`: adjusted `SIMULATION_FUNCTION_EXPLANATION` and related builder outputs for clearer guidance to the model.
- `agent_test_gpt/toolchain.py`: `format_simulation_tools` now returns an empty list (simulation tools are internal), and `normalize_tool_chain` drops explicit `run_ngspice` entries while injecting required simulation steps for analyses.
- Tests updated to match the new behavior:
  - `tests/test_prompts.py`: updated expected SHA256 values to match current template text.
  - `tests/test_tool_functions.py`: updated expectation that `format_simulation_tools` returns an empty list.

Validation performed
- Ran unit tests: `python3 -m unittest discover -s tests -p 'test_*.py' -v` — all tests passed after updates.
- Ran main agent: `python3 -m agent_test_gpt.agent_gpt_openai` with `OPENAI_API_KEY` exported; the run made multiple streaming LLM calls and NGspice runs, wrote outputs to `output/runs/<id>/`, and progressed through several optimization iterations before being cancelled manually during a long streaming response.

Notes
- These changes are intentional; if you prefer the previous behavior (explicit `run_ngspice` tool calls from the model), indicate and I will revert the toolchain/tests accordingly.

Files touched
- `agent_test_gpt/prompts.py`
- `agent_test_gpt/toolchain.py`
- `tests/test_prompts.py`
- `tests/test_tool_functions.py`

