import json
from pathlib import Path

import pytest

from agent_test_gpt.optimization import (
    OptimizationConfig,
    OptimizationContext,
    OptimizationDeps,
    OptimizationRunner,
)


class _LLMMock:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt: str):
        self.calls.append(prompt)
        return f"resp-{len(self.calls)}"


def test_optimization_runner_converges_with_mocked_deps(tmp_path: Path):
    # Build a minimal tool chain: simulate dc, run ngspice, then compute ac_gain.
    tool_chain = {
        "tool_calls": [
            {"name": "dc_simulation"},
            {"name": "run_ngspice"},
            {"name": "ac_gain"},
        ]
    }

    # No explicit targets -> all passes start True
    target_values = json.dumps({"target_values": [{}]})

    llm = _LLMMock()

    def _run_ngspice_mock(_net, _name, output_dir, **_kwargs):
        (Path(output_dir) / "output_dc.dat").write_text("0 0\n")
        return True

    deps = OptimizationDeps(
        make_chat_completion_request=llm,
        sanitize_netlist=lambda x: x,
        dc_simulation=lambda net, *_args, **_kwargs: net + "\n.dc",
        ac_simulation=lambda net, *_args, **_kwargs: net + "\n.ac",
        trans_simulation=lambda net, *_args, **_kwargs: net + "\n.tran",
        run_ngspice=_run_ngspice_mock,
        filter_lines=lambda *_args, **_kwargs: None,
        convert_to_csv=lambda *_args, **_kwargs: True,
        format_csv_to_key_value=lambda *_args, **_kwargs: None,
        read_txt_as_string=lambda *_args, **_kwargs: "No values found where vgs - vth < 0.",
        ac_gain=lambda *_args, **_kwargs: 10.0,
        dc_gain=lambda *_args, **_kwargs: None,
        out_swing=lambda *_args, **_kwargs: None,
        offset=lambda *_args, **_kwargs: None,
        ICMR=lambda *_args, **_kwargs: None,
        tran_gain=lambda *_args, **_kwargs: None,
        bandwidth=lambda *_args, **_kwargs: None,
        unity_bandwidth=lambda *_args, **_kwargs: None,
        phase_margin=lambda *_args, **_kwargs: None,
        stat_power=lambda *_args, **_kwargs: None,
        thd_input_range=lambda *_args, **_kwargs: (None, None),
        cmrr_tran=lambda *_args, **_kwargs: (None, None),
        extract_code=lambda text: text,
    )

    context = OptimizationContext(
        sim_output="sim_output",
        sizing_question="size me",
        type_identified="type",
        source_names=["Vin"],
        output_nodes=["out"],
        output_dir=str(tmp_path),
    )
    config = OptimizationConfig.from_output_dir(str(tmp_path))

    runner = OptimizationRunner(context, deps, config)
    initial_metrics = {
        "gain_output": None,
        "tr_gain_output": None,
        "output_swing_output": None,
        "input_offset_output": None,
        "icmr_output": None,
        "ubw_output": None,
        "pm_output": None,
        "pr_output": None,
        "cmrr_output": None,
        "thd_output": None,
    }
    result, opti_netlist = runner.run(
        tool_chain,
        target_values,
        "netlist",
        lambda x: float(str(x).replace("dB", "").replace("Hz", "")) if x is not None else None,
        initial_metrics,
    )

    assert result["converged"] is True
    assert result["gain_output"] == 10.0
    assert opti_netlist.endswith("\n.dc")
    # Ensure LLM was called for analysis, optimising, sizing prompts
    assert len(llm.calls) == 3
