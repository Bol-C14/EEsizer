# `agent_gpt_openai.ipynb` Flow Notes

## Overview
The notebook orchestrates an LLM-in-the-loop analog-circuit optimization loop. It starts by defining the user's objective and a PTM-90nm CMOS netlist, asks GPT-based prompts to decompose tasks and extract targets, identifies circuit metadata (type, I/O nodes), plans the required simulations, runs ngspice plus Python post-processing utilities, and finally launches an iterative LLM-guided sizing loop before persisting reports/plots.【F:agent_test_gpt/agent_gpt_openai.py†L2-L1977】

## External dependencies & files
- **Process model**: the base netlist references `ptm_90.txt` via a SPICE `.include` directive, so that model file must exist relative to the notebook working dir.【F:agent_test_gpt/agent_gpt_openai.py†L158-L210】
- **Simulation binaries**: `run_ngspice` shells out to `/home/chang/Documents/test/ngspice-44.2/release/src/ngspice`, writing transient outputs to `output/*.dat` plus `output/op.txt`. Ensure this binary path is reachable or update it before running.【F:agent_test_gpt/agent_gpt_openai.py†L1231-L1245】
- **Working directories**: tool helpers expect `output/`, `output/90nm/`, and nested folders such as `output/90nm/netlist_cs_o3/` to exist (or be creatable) because netlists, CSV logs, history text files, and Matplotlib PDFs land there.【F:agent_test_gpt/agent_gpt_openai.py†L1231-L1976】

## Cell-by-cell inventory
| Cell # | Type | Purpose | Key inputs | Outputs / side effects | External artifacts |
| --- | --- | --- | --- | --- | --- |
| 0 | Code | Imports numpy/matplotlib/OpenAI tooling plus FFT helpers. | Python stdlib + libs | Module-level imports | – |【F:agent_test_gpt/agent_gpt_openai.py†L2-L16】
| 1 | Code | Loads `.env` via `dotenv`, pulls `OPENAI_API_KEY`. | `.env` file | Populates `api_key` | `.env` |【F:agent_test_gpt/agent_gpt_openai.py†L19-L30】
| 2 | Code | Defines `make_chat_completion_request` (streaming chat completion). | Prompt string | Generated text string | OpenAI API |【F:agent_test_gpt/agent_gpt_openai.py†L33-L76】
| 3 | Code | Defines `make_chat_completion_request_function` (tool-calling schema). | Prompt + tool spec | OpenAI response object | OpenAI API |【F:agent_test_gpt/agent_gpt_openai.py†L79-L152】
| 4 | Markdown | Section header “User Input”. | – | – | – |【F:agent_test_gpt/agent_gpt_openai.py†L154-L155】
| 5 | Code | Declares natural-language goal & PTM-90 CMOS netlist string. | Hard-coded string | `tasks_generation_question`, `netlist` | References `ptm_90.txt` |【F:agent_test_gpt/agent_gpt_openai.py†L158-L210】
| 6 | Markdown | Section header “Task Decomposition”. | – | – | – |【F:agent_test_gpt/agent_gpt_openai.py†L213-L215】
| 7 | Code | Builds `tasks_generation_template` prompt describing JSON decomposition format. | Example question/netlist | Template string | – |【F:agent_test_gpt/agent_gpt_openai.py†L217-L245】
| 8 | Code | Formats `tasks_generation_prompt`, calls GPT, stores raw JSON-like string in `tasks`. | Template + question + netlist | `tasks` text | – |【F:agent_test_gpt/agent_gpt_openai.py†L247-L255】
| 9 | Code | Defines `get_tasks` parser stripping `````/`json` noise and populates globals `type_question`, `node_question`, `sim_question`, `sizing_question`. | `tasks` string | Global question strings | – |【F:agent_test_gpt/agent_gpt_openai.py†L257-L285】
| 10 | Code | Creates `target_value_SYSTEM_PROMPT`, assembles prompt, asks GPT for `{ "target_values": [...] }`. | `tasks_generation_question` | `target_values` text | – |【F:agent_test_gpt/agent_gpt_openai.py†L288-L317】
| 11 | Code | Defines circuit-type identification template/prompt. | `netlist` | `type_identify_prompt` | – |【F:agent_test_gpt/agent_gpt_openai.py†L321-L326】
| 12 | Code | Calls GPT to classify the circuit type. | `type_identify_prompt` | `type_identified` text | – |【F:agent_test_gpt/agent_gpt_openai.py†L328-L330】
| 13 | Code | Declares `nodes_extract`/`extract_code` helpers plus `node_SYSTEM_PROMPT` and prompt builder. | Node question + netlist | Functions & prompt strings | – |【F:agent_test_gpt/agent_gpt_openai.py†L333-L404】
| 14 | Code | Sends `node_prompt` to GPT to retrieve JSON node metadata. | Node prompt | `nodes` string | – |【F:agent_test_gpt/agent_gpt_openai.py†L406-L409】
| 15 | Code | Parses node JSON into `input_nodes`, `output_nodes`, `source_names`. | `nodes` string | Python lists | – |【F:agent_test_gpt/agent_gpt_openai.py†L411-L415】
| 16 | Code | Declares netlist augmentation helpers (`dc_simulation`, `ac_simulation`, `trans_simulation`, `tran_inrange`). | Netlist string + node names | Modified netlists with `.control` blocks | Writes to `output/output_*.dat` when later run |【F:agent_test_gpt/agent_gpt_openai.py†L418-L494】
| 17 | Code | Imports `find_peaks` and seeds globals for initial spec measurements; defines `out_swing`, `offset`, `ICMR`, etc. | – | Measurement utilities | Read/write `output/netlist.cir`, `output/output_*.dat` |【F:agent_test_gpt/agent_gpt_openai.py†L495-L740】
| 18 | Code | Defines `simulation_function_explanation` instructions describing how to select sim types/tools/analyses. | – | Prompt template string | – |【F:agent_test_gpt/agent_gpt_openai.py†L1359-L1372】
| 19 | Code | Prints `sim_question` (from task decomposition). | `sim_question` | Console output | – |【F:agent_test_gpt/agent_gpt_openai.py†L1374-L1376】
| 20 | Code | Prints `sizing_question`. | `sizing_question` | Console output | – |【F:agent_test_gpt/agent_gpt_openai.py†L1377-L1378】
| 21 | Code | Formats `sim_prompts_gen` combining explanation + `sizing_question`. | Strings | Printed prompt | – |【F:agent_test_gpt/agent_gpt_openai.py†L1380-L1386】
| 22 | Code | Calls `make_chat_completion_request_function(sizing_question)` to get tool selections. | `sizing_question` | OpenAI response (`tool`) | – |【F:agent_test_gpt/agent_gpt_openai.py†L1388-L1390】
| 23 | Code | Defines helpers to parse tool calls, deduplicate sim types/tools, order analyses, combine into `tool_chain`. | `tool` response | `tool_chain` dict | – |【F:agent_test_gpt/agent_gpt_openai.py†L1392-L1487】
| 24 | Code | Prints tool chain, runs `tool_calling(tool_chain)` to execute sims and compute initial specs (populating globals and returning `sim_output`, `sim_netlist`). | `tool_chain` | Measurement string + augmented netlist | Writes numerous `output/*.dat`, `output/vgscheck.*`, `output/netlist.cir` |【F:agent_test_gpt/agent_gpt_openai.py†L1247-L1346】【F:agent_test_gpt/agent_gpt_openai.py†L1488-L1497】
| 25 | Markdown | “Optimization” section header. | – | – | – |【F:agent_test_gpt/agent_gpt_openai.py†L1499-L1501】
| 26 | Code | Builds `analysing_system_prompt` (step-by-step relationship reasoning). | – | Prompt string | – |【F:agent_test_gpt/agent_gpt_openai.py†L1503-L1520】
| 27 | Code | Creates `sizing_Question = f"Currently, {sim_output}. " + sizing_question`. | `sim_output`, `sizing_question` | Combined question text | – |【F:agent_test_gpt/agent_gpt_openai.py†L1523-L1525】
| 28 | Code | Defines `optimising_system_prompt` (guidelines for generating optimization suggestions). | – | Prompt string | – |【F:agent_test_gpt/agent_gpt_openai.py†L1528-L1543】
| 29 | Code | Declares `sizing_SYSTEM_PROMPT` plus `sizing_output_template` that constrains device edits. | – | Prompt + constraint text | – |【F:agent_test_gpt/agent_gpt_openai.py†L1545-L1563】
| 30 | Code | Implements `optimization(...)`, the main iterative loop performing LLM analysis, optimization, sizing, simulation reruns, convergence checks, history logging, and CSV exports. | `tool_chain`, `target_values`, `sim_netlist` | Result dict + optimized netlist; writes `output/90nm/result_history.txt`, `output/90nm/g2_o3.csv` | Multiple files inside `output/` tree |【F:agent_test_gpt/agent_gpt_openai.py†L1566-L1897】
| 31 | Code | Wraps optimizer via `run_multiple_optimizations` and executes it (`num_runs=1`). | `target_values`, `sim_netlist` | `results` list | – |【F:agent_test_gpt/agent_gpt_openai.py†L1900-L1909】
| 32 | Code | Copies final netlist from `output/netlist.cir` to `output/90nm/netlist_cs_o3/a1.cir`. | File paths | Duplicated netlist | File copy |【F:agent_test_gpt/agent_gpt_openai.py†L1912-L1917】
| 33 | Code | Loads `output/90nm/g2_o3.csv`, derives batches, plots nine metrics over iterations, saves `railtorail_subplots_4x2_g1.pdf`. | CSV log | Matplotlib figure/PDF | `output/90nm/g2_o3.csv`, `railtorail_subplots_4x2_g1.pdf` |【F:agent_test_gpt/agent_gpt_openai.py†L1919-L1976】

## Data flow narrative
1. **Task metadata extraction (Cells 0‑15)**
   - After defining the goal and the transistor-level netlist, GPT decomposes the work into four question strings (type/node/simulation/sizing) and extracts numeric targets as JSON. Additional GPT calls classify the circuit and enumerate node/source names, which are parsed into Python lists for downstream netlist augmentation.【F:agent_test_gpt/agent_gpt_openai.py†L158-L415】
2. **Simulation planning and execution (Cells 16‑24)**
   - Netlist mutators (`dc_simulation`, `ac_simulation`, `trans_simulation`, `tran_inrange`) splice `.control` decks that write ngspice `.dat` dumps. Measurement helpers compute gain, swing, offsets, etc., mostly by reading `output/netlist.cir` and the resulting data files. The `simulation_function_explanation` prompt tells GPT how to pick simulation/analysis steps; the response is parsed into `tool_chain`, which `tool_calling` iterates to (a) mutate the working netlist, (b) run ngspice once, and (c) call the requested measurement functions, finally constructing the `sim_output` summary string.【F:agent_test_gpt/agent_gpt_openai.py†L418-L1346】【F:agent_test_gpt/agent_gpt_openai.py†L1359-L1497】
3. **Optimization loop (Cells 25‑32)**
   - Prompt templates (`analysing_system_prompt`, `optimising_system_prompt`, `sizing_SYSTEM_PROMPT`, `sizing_output_template`) define a multi-step workflow: analyze parameter/signal deltas, suggest edits, and apply constrained sizing changes. The `optimization` function replays this trio each iteration, extracts the updated netlist from triple-quoted LLM responses, re-runs the same `tool_chain`, compares results against target specs (with ±5% tolerance), logs history, and stops once every metric (plus Vgs-Vth checks) passes. Outputs include CSV logs, history text, and the latest SPICE deck copied into a process-specific folder. Finally, a Matplotlib routine summarizes metrics per iteration and exports a PDF for reporting.【F:agent_test_gpt/agent_gpt_openai.py†L1503-L1976】

## Prompt templates (quick reference)
- **Task decomposition** – `tasks_generation_template` expects JSON with `type_question`, `node_question`, `sim_question`, `sizing_question`. Use the provided example when adjusting instructions.【F:agent_test_gpt/agent_gpt_openai.py†L217-L245】
- **Target extraction** – `target_value_SYSTEM_PROMPT` enforces `{ "target_values": [...] }`, frequency-unit normalization, and sign preservation.【F:agent_test_gpt/agent_gpt_openai.py†L288-L317】
- **Circuit classification** – `type_identify_template` simply states “identify the circuit type”.【F:agent_test_gpt/agent_gpt_openai.py†L321-L326】
- **Node detection** – `node_SYSTEM_PROMPT` requests a bare JSON dictionary listing `input_node`, `output_node`, `source_name`. This powers netlist augmentation logic.【F:agent_test_gpt/agent_gpt_openai.py†L392-L404】
- **Simulation planning** – `simulation_function_explanation` instructs the tool to enumerate simulation types, the `run_ngspice` tool, and per-metric analysis functions, with caution about `cmrr_tran`/`thd_input_range`. This enables automated construction of the `tool_chain`.【F:agent_test_gpt/agent_gpt_openai.py†L1359-L1372】【F:agent_test_gpt/agent_gpt_openai.py†L1392-L1487】
- **Optimization trio** – `analysing_system_prompt`, `optimising_system_prompt`, `sizing_SYSTEM_PROMPT`, and `sizing_output_template` encode the behavior expected from each reasoning stage and the guardrails for editing transistor sizes/biases.【F:agent_test_gpt/agent_gpt_openai.py†L1503-L1563】

## Simulation & optimization execution order
1. Build `tool_chain` (`format_*` helpers) from GPT function-call arguments.【F:agent_test_gpt/agent_gpt_openai.py†L1392-L1484】
2. Run `tool_calling(tool_chain)`:
   - sequentially decorate the base netlist with every simulation block, run ngspice once, then call all measurement functions (AC gain, output swing, THD, etc.) to populate global `*_init` variables and the `sim_output` string.【F:agent_test_gpt/agent_gpt_openai.py†L1231-L1346】
3. Launch `run_multiple_optimizations`, which calls `optimization` exactly once (default). Inside each iteration: analysis prompt → optimization prompt → sizing prompt → netlist extraction → ngspice run → metric computation → tolerance check → history persistence → repeat until success or `max_iterations` (25).【F:agent_test_gpt/agent_gpt_openai.py†L1566-L1909】
4. After convergence, copy `output/netlist.cir` to the process-specific folder and regenerate plots from `output/90nm/g2_o3.csv`.【F:agent_test_gpt/agent_gpt_openai.py†L1912-L1976】

## Data artifacts & logs
- **Netlists**: `output/netlist.cir` (latest sim deck) and `output/90nm/netlist_cs_o3/a1.cir` (archived copy).【F:agent_test_gpt/agent_gpt_openai.py†L1231-L1346】【F:agent_test_gpt/agent_gpt_openai.py†L1912-L1917】
- **Simulation dumps**: `output/output_dc.dat`, `output/output_ac.dat`, `output/output_tran.dat`, `output/output_tran_inrange.dat`, `output/output_inrange_cmrr.dat`, etc., produced by `.control` decks inside the generated netlists.【F:agent_test_gpt/agent_gpt_openai.py†L418-L702】【F:agent_test_gpt/agent_gpt_openai.py†L800-L1159】
- **Constraint checks**: `output/op.txt`, `output/vgscheck.txt`, `output/vgscheck.csv`, `output/vgscheck_output.txt` capture ngspice operating-point data and derived Vgs-Vth violations.【F:agent_test_gpt/agent_gpt_openai.py†L1231-L1315】
- **Iteration history**: `output/90nm/result_history.txt` (rolling text) and `output/90nm/g2_o3.csv` (structured metrics). The CSV is also the basis for the final Matplotlib dashboards, saved as `railtorail_subplots_4x2_g1.pdf`.【F:agent_test_gpt/agent_gpt_openai.py†L1585-L1976】

Use this document as a map when refactoring: follow the cell inventory to locate logic, consult the prompt list to understand LLM expectations, and trace the execution order plus artifacts to keep the pipeline consistent.
