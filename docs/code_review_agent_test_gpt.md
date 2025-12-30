# Code Review: `agent_test_gpt`

Scope: current `agent_test_gpt` package after the initial refactors (LLM prompts, tool calling, optimization, reporting). Findings are ordered roughly by severity. Line numbers reference the current tree.

## Critical
- Unbounded, untrusted code execution via LLM-controlled inputs: the main driver feeds raw LLM JSON into `globals()` (`agent_test_gpt/agent_gpt_openai.py:159-177`) and later passes LLM-crafted netlists to `run_ngspice` (`agent_test_gpt/agent_gpt_openai.py:351`, `agent_test_gpt/optimization.py:351`), giving the model effective code execution and filesystem write ability with no validation or sandboxing.
- Mitigation: `agent_test_gpt/agent_gpt_openai.py` now parses LLM JSON into typed dataclasses (`TaskQuestions`) without touching globals; netlists are sanitized (`sanitize_netlist`) to strip `.control` blocks and unsafe includes, and every simulation/optimization run is scoped to a per-run output dir with validated ngspice invocation.
- Unsafe argument parsing with `eval`: `extract_tool_data` falls back to `eval` on LLM-supplied strings (`agent_test_gpt/toolchain.py:70-74`), enabling arbitrary code execution.
- Mitigation: `extract_tool_data` now uses strict JSON parsing only, supports concatenated objects without `eval`, and drops unknown argument types.
- Hard-coded user input and netlist: a single fixed question and circuit string are baked into the entrypoint (`agent_test_gpt/agent_gpt_openai.py:78-144`), so the app cannot be safely reused across users, and these constants are reused in optimization/reporting paths without override hooks.
- Mitigation: `run_agent(user_question, user_netlist, run_dir)` accepts caller-provided inputs; defaults remain only for compatibility, with per-run isolation.
- LLM-defined globals and variable injection: `get_tasks` writes arbitrary keys from LLM output into module globals (`agent_test_gpt/agent_gpt_openai.py:159-177`) and `_parse_target_values` repeats the pattern (`agent_test_gpt/optimization.py:229-290`), letting the model define variables/code paths and creating cross-run leakage.
- Mitigation: globals are removed; task/target parsing returns structured objects (`TaskQuestions`, `TargetValues`) and flows carry state explicitly through contexts.
- Tool/optimization loops run external processes without timeouts or error gates: `run_ngspice` spawns ngspice on unvalidated netlists with no timeout or resource guard (`agent_test_gpt/simulation_utils.py:777-844`). Optimization repeats LLM calls and ngspice invocations inside a while-loop gated only by iteration count and sleeps (`agent_test_gpt/optimization.py:521-599`), risking runaway jobs.
- Mitigation: `run_ngspice` now enforces timeouts and per-run output directories; tool/optimization runners raise on failed ngspice runs. Fixed sleeps were removed in favor of configurable `llm_delay_seconds` (default 0) so loops do not stall silently.

## Major
- Global state mutation driven by LLM: both task parsing and target parsing write into module globals (`agent_test_gpt/agent_gpt_openai.py:159-177`, `agent_test_gpt/optimization.py:229-290`), making behavior order-dependent, non-reentrant, and hard to test.
- Missing validation of tool chains: `ToolChainRunner.run` trusts the tool list and only dispatches by name (`agent_test_gpt/optimization.py:102-177`). Missing or out-of-order steps (e.g., calling `ac_gain` before `run_ngspice`) will raise file errors, and invalid names are silently ignored.
- Fragile, hard-coded file system layout: most helpers assume fixed relative paths like `output/netlist.cir` and `output/op.txt` (`agent_test_gpt/simulation_utils.py:119-178`, `210-232`, `777-844`; `agent_test_gpt/config.py:3-13`), so concurrent runs or different working directories corrupt each other. No temp directories or per-run isolation.
- Error handling is largely prints-or-nothing: LLM client swallows exceptions and returns `None` (`agent_test_gpt/llm_client.py:33-60`, `127-145`), simulation helpers print and proceed, and optimization uses no try/except around LLM or ngspice calls (`agent_test_gpt/optimization.py:312-376`, `521-599`), so failures propagate as confusing follow-on errors.
- Inconsistent/incorrect metric logic: `offset` can reference `voff` before assignment when the data slice is empty (`agent_test_gpt/simulation_utils.py:224-228`). `_parse_target_values` negates THD targets unconditionally and mixes pass flag defaults (`agent_test_gpt/optimization.py:229-290`), risking false positives/negatives.
- Metric helpers rewrite and reuse global files: many functions read/write `output/netlist.cir` and derived `output/*.dat` regardless of the caller’s netlist (`agent_test_gpt/simulation_utils.py:119-232`, `485-549`, `551-642`), so concurrent runs trample each other and the results depend on ambient files, not inputs.
- Missing existence checks before parsing data: helpers call `np.genfromtxt` on hard-coded paths with no guards (e.g., `agent_test_gpt/simulation_utils.py:119-232`, `335-354`, `356-417`, `420-449`), so a failed ngspice run crashes later with opaque tracebacks.

## Moderate
- Tight coupling to specific LLM models and settings: model names and temperatures are hard-coded (`agent_test_gpt/llm_client.py:33-47`, `127-136`), with no API key wiring or retry/backoff. This hinders provider changes and observability.
- Sleep-based pacing instead of structured retries: fixed `time.sleep(10/20)` calls gate every loop (`agent_test_gpt/optimization.py:312-326`, `521-533`), hurting throughput and leaving no adaptive backoff on failure.
- Metric and tool naming drift: simulation/analysis naming is hand-built and partially inferred (`agent_test_gpt/toolchain.py:86-205`), making it easy for LLM responses to fall out of sync with actual tool implementations.
- Reporting assumes CSVs exist and are well-formed: `plot_optimization_history` reads CSV without guardrails and uses derived columns directly (`agent_test_gpt/reporting.py:52-92`), so missing files or malformed data will crash plotting.
- Reproducibility gaps: runs depend on ambient cwd, existing `output` contents, environment variables for ngspice, and whatever the LLM returns. There is no seed/control for randomness or snapshotting of prompts/responses beyond ad-hoc prints.
- Numerical correctness issues: `cmrr_tran` uses natural log instead of log10 for dB (`agent_test_gpt/simulation_utils.py:533-548`), `stat_power` indexes column 3 even when only 3 columns exist (`agent_test_gpt/simulation_utils.py:466-482`), and several AC helpers take `log10` of complex arrays, relying on numpy coercion/warnings rather than explicit magnitude (`agent_test_gpt/simulation_utils.py:356-417`).

## Minor / Style
- Mixed casing and unused variables: duplicated `import os` (`agent_test_gpt/agent_gpt_openai.py:3` and 59), placeholder `_missing_dc_gain` injected instead of a real metric (`agent_test_gpt/agent_gpt_openai.py:336-347`), and unused `type_question` defaults (`agent_test_gpt/agent_gpt_openai.py:156-224`).
- Prints instead of structured logging throughout, making it hard to trace multi-run execution or capture logs programmatically.
- Tests cover only prompt hashing and a few parsing helpers; there is no coverage for optimization loops, ngspice orchestration, or safety constraints.

## Recommendations / Next Steps
- Eliminate unsafe parsing: remove `eval`, require strict JSON schemas for tool calls, and validate against an allowlist before dispatch.
- Remove LLM control over globals and file paths: pass structured data through typed objects, scope state per run, and isolate all file outputs to per-run temp dirs.
- Introduce input validation and guardrails before running ngspice: sanitize netlists, enforce max size/runtime, and add timeouts with clear error propagation.
- Decouple configuration: move model names, paths, and iteration/timing controls into a configuration layer with sane defaults and environment overrides.
- Add structured logging, retry/backoff for LLM calls, and fast-fail error handling around simulation and optimization steps.
- Expand tests to cover tool-chain dispatch, failure modes (missing files, bad LLM JSON), and safety constraints (no eval, bounded processes), ideally with ngspice mocked.
