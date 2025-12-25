import ast
import contextlib
import io
import json
import re
import types
import unittest
from pathlib import Path


def load_symbols(function_names, assign_names=None):
    assign_names = set(assign_names or [])
    source_path = Path(__file__).resolve().parents[1] / "agent_test_gpt" / "agent_gpt_openai.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_nodes = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assign_names:
                    selected_nodes.append(node)
                    break
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in assign_names:
                selected_nodes.append(node)

    module = types.SimpleNamespace()
    globals_dict = {
        "__name__": "agent_tools_test",
        "Path": Path,
        "json": json,
        "re": re,
    }
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec")
    exec(compiled, globals_dict)

    for name in function_names:
        setattr(module, name, globals_dict[name])
    for name in assign_names:
        setattr(module, name, globals_dict[name])

    return module


FUNCTIONS = [
    "_resolve_spice_include_path",
    "normalize_spice_includes",
    "nodes_extract",
    "extract_code",
    "extract_number",
    "extract_tool_data",
    "_infer_simulation_type_from_analysis",
    "format_simulation_types",
    "format_simulation_tools",
    "format_analysis_types",
    "combine_results",
]
ASSIGNS = ["_INCLUDE_PATTERN"]

SYMBOLS = load_symbols(FUNCTIONS, ASSIGNS)


class TestSpiceIncludeHelpers(unittest.TestCase):
    def test_resolve_spice_include_path_absolute(self):
        candidate = Path("agent_test_gpt/ptm_90.txt").resolve()
        resolved = SYMBOLS._resolve_spice_include_path(str(candidate))
        self.assertEqual(resolved, str(candidate))

    def test_resolve_spice_include_path_search_roots(self):
        resolved = SYMBOLS._resolve_spice_include_path("ptm_90.txt")
        self.assertEqual(resolved, "agent_test_gpt/ptm_90.txt")

    def test_normalize_spice_includes(self):
        netlist = ".include 'ptm_90.txt'\nR1 out 0 1k\n"
        expected = ".include 'agent_test_gpt/ptm_90.txt'\nR1 out 0 1k\n"
        self.assertEqual(SYMBOLS.normalize_spice_includes(netlist), expected)


class TestNodeAndCodeExtractors(unittest.TestCase):
    def test_nodes_extract_legacy_schema(self):
        payload = '{"nodes": [{"input_node": "in1"}, {"output_node": "out"}, {"source_name": "Vid"}]}'
        input_nodes, output_nodes, source_names = SYMBOLS.nodes_extract(payload)
        self.assertEqual(input_nodes, ["in1"])
        self.assertEqual(output_nodes, ["out"])
        self.assertEqual(source_names, ["Vid"])

    def test_nodes_extract_new_schema(self):
        payload = '{"nodes": [{"input_nodes": ["in1", "in2"]}, {"source_names": ["Vid", "Vcm"]}]}'
        input_nodes, output_nodes, source_names = SYMBOLS.nodes_extract(payload)
        self.assertEqual(input_nodes, ["in1", "in2"])
        self.assertEqual(output_nodes, [])
        self.assertEqual(source_names, ["Vid", "Vcm"])

    def test_nodes_extract_invalid_json(self):
        with contextlib.redirect_stdout(io.StringIO()):
            input_nodes, output_nodes, source_names = SYMBOLS.nodes_extract("not json")
        self.assertEqual(input_nodes, [])
        self.assertEqual(output_nodes, [])
        self.assertEqual(source_names, [])

    def test_extract_code_blocks(self):
        text = "Intro '''line1\nline2\n''' middle ```line3\nline4``` tail"
        extracted = SYMBOLS.extract_code(text)
        self.assertEqual(extracted, "line1\nline2\nline3\nline4")


class TestToolParsing(unittest.TestCase):
    class FakeFunction:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, function):
            self.function = function

    class FakeMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, choices):
            self.choices = choices

    def test_extract_tool_data_various_arguments(self):
        f1 = self.FakeFunction(
            name="universal_circuit_tool",
            arguments='{"simulation_type": "ac", "analysis_type": "ac_gain", "simulation_tool": "run_ngspice"}',
        )
        f2 = self.FakeFunction(
            name="universal_circuit_tool",
            arguments='{"analysis_type": "tran_gain"}',
        )
        f3 = self.FakeFunction(
            name="universal_circuit_tool",
            arguments="",
        )
        f4 = self.FakeFunction(
            name="universal_circuit_tool",
            arguments='{"analysis_type":"ac_gain"}{"analysis_type":"phase_margin"}',
        )

        tool_calls = [self.FakeToolCall(f) for f in [f1, f2, f3, f4]]
        message = self.FakeMessage(tool_calls=tool_calls)
        response = self.FakeResponse(choices=[self.FakeChoice(message=message)])

        tool_data_list = SYMBOLS.extract_tool_data(response)
        self.assertEqual(len(tool_data_list), 4)
        self.assertEqual(tool_data_list[0]["simulation_type"], "ac")
        self.assertEqual(tool_data_list[0]["analysis_type"], "ac_gain")
        self.assertEqual(tool_data_list[0]["simulation_tool"], "run_ngspice")
        self.assertEqual(tool_data_list[1]["analysis_type"], "tran_gain")
        self.assertEqual(tool_data_list[1]["simulation_tool"], "run_ngspice")
        self.assertEqual(tool_data_list[2]["analysis_type"], "ac_gain")
        self.assertEqual(tool_data_list[3]["analysis_type"], "phase_margin")


class TestToolFormatting(unittest.TestCase):
    def test_infer_simulation_type(self):
        self.assertEqual(SYMBOLS._infer_simulation_type_from_analysis("ac_gain"), "ac")
        self.assertEqual(SYMBOLS._infer_simulation_type_from_analysis(["tran_gain"]), "transient")
        self.assertEqual(SYMBOLS._infer_simulation_type_from_analysis("output_swing"), "dc")
        self.assertIsNone(SYMBOLS._infer_simulation_type_from_analysis("unknown"))

    def test_format_simulation_types(self):
        tool_data_list = [
            {"analysis_type": "ac_gain"},
            {"analysis_type": "tran_gain"},
            {"analysis_type": "output_swing"},
        ]
        expected = [
            {"name": "ac_simulation"},
            {"name": "transient_simulation"},
            {"name": "dc_simulation"},
        ]
        self.assertEqual(SYMBOLS.format_simulation_types(tool_data_list), expected)

    def test_format_simulation_tools(self):
        tool_data_list = [
            {"simulation_tool": "run_ngspice"},
            {"raw_args": {}},
            {},
        ]
        expected = [{"name": "run_ngspice"}]
        self.assertEqual(SYMBOLS.format_simulation_tools(tool_data_list), expected)

    def test_format_analysis_types(self):
        tool_data_list = [
            {"analysis_type": "ac_gain"},
            {"analysis_type": ["cmrr_tran", "thd_input_range", "phase_margin"]},
            {"simulation_type": "transient"},
        ]
        expected = [
            {"name": "ac_gain"},
            {"name": "phase_margin"},
            {"name": "tran_gain"},
            {"name": "cmrr_tran"},
            {"name": "thd_input_range"},
        ]
        self.assertEqual(SYMBOLS.format_analysis_types(tool_data_list), expected)

    def test_combine_results(self):
        sim_types = [{"name": "ac_simulation"}]
        sim_tools = [{"name": "run_ngspice"}]
        analysis = [{"name": "ac_gain"}]
        self.assertEqual(
            SYMBOLS.combine_results(sim_types, sim_tools, analysis),
            sim_types + sim_tools + analysis,
        )

    def test_extract_number(self):
        self.assertEqual(SYMBOLS.extract_number("gain=-3.5dB"), -3.5)
        self.assertEqual(SYMBOLS.extract_number("10 MHz"), 10.0)
        self.assertIsNone(SYMBOLS.extract_number("no number"))


if __name__ == "__main__":
    unittest.main()
