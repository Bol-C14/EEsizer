import unittest
from pathlib import Path

import pytest

from eesizer_core.baselines.legacy_metrics_adapter import ensure_legacy_importable

if not ensure_legacy_importable():
    pytest.skip("legacy_eesizer not available", allow_module_level=True)

from legacy_eesizer import netlist_utils
from legacy_eesizer import toolchain


class TestSpiceIncludeHelpers(unittest.TestCase):
    def test_resolve_spice_include_path_absolute(self):
        candidate = Path("legacy/legacy_eesizer/resources/ptm_90.txt").resolve()
        resolved = netlist_utils._resolve_spice_include_path(str(candidate))
        self.assertEqual(resolved, str(candidate))

    def test_resolve_spice_include_path_search_roots(self):
        resolved = netlist_utils._resolve_spice_include_path("ptm_90.txt")
        self.assertEqual(resolved, "legacy/legacy_eesizer/resources/ptm_90.txt")

    def test_normalize_spice_includes(self):
        netlist = ".include 'ptm_90.txt'\nR1 out 0 1k\n"
        expected = ".include 'legacy/legacy_eesizer/resources/ptm_90.txt'\nR1 out 0 1k\n"
        self.assertEqual(netlist_utils.normalize_spice_includes(netlist), expected)


class TestNodeAndCodeExtractors(unittest.TestCase):
    def test_nodes_extract_legacy_schema(self):
        payload = '{"nodes": [{"input_node": "in1"}, {"output_node": "out"}, {"source_name": "Vid"}]}'
        input_nodes, output_nodes, source_names = netlist_utils.nodes_extract(payload)
        self.assertEqual(input_nodes, ["in1"])
        self.assertEqual(output_nodes, ["out"])
        self.assertEqual(source_names, ["Vid"])

    def test_nodes_extract_new_schema(self):
        payload = '{"nodes": [{"input_nodes": ["in1", "in2"], "output_nodes": ["out"], "source_names": ["Vid", "Vcm"]}]}'
        input_nodes, output_nodes, source_names = netlist_utils.nodes_extract(payload)
        self.assertEqual(input_nodes, ["in1", "in2"])
        self.assertEqual(output_nodes, ["out"])
        self.assertEqual(source_names, ["Vid", "Vcm"])

    def test_nodes_extract_invalid_json(self):
        with self.assertRaises(Exception):
            netlist_utils.nodes_extract("not json")

    def test_extract_code_blocks(self):
        text = "Intro '''line1\nline2\n''' middle ```line3\nline4``` tail"
        extracted = netlist_utils.extract_code(text)
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

        tool_data_list = toolchain.extract_tool_data(response)
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
        self.assertEqual(toolchain._infer_simulation_type_from_analysis("ac_gain"), "ac")
        self.assertEqual(toolchain._infer_simulation_type_from_analysis(["tran_gain"]), "transient")
        self.assertEqual(toolchain._infer_simulation_type_from_analysis("output_swing"), "dc")
        self.assertIsNone(toolchain._infer_simulation_type_from_analysis("unknown"))

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
        self.assertEqual(toolchain.format_simulation_types(tool_data_list), expected)

    def test_format_simulation_tools(self):
        tool_data_list = [
            {"simulation_tool": "run_ngspice"},
            {"raw_args": {}},
            {},
        ]
        expected = []
        self.assertEqual(toolchain.format_simulation_tools(tool_data_list), expected)

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
        self.assertEqual(toolchain.format_analysis_types(tool_data_list), expected)

    def test_combine_results(self):
        sim_types = [{"name": "ac_simulation"}]
        sim_tools = [{"name": "run_ngspice"}]
        analysis = [{"name": "ac_gain"}]
        self.assertEqual(
            toolchain.combine_results(sim_types, sim_tools, analysis),
            sim_types + sim_tools + analysis,
        )

    def test_extract_number(self):
        self.assertEqual(netlist_utils.extract_number("gain=-3.5dB"), -3.5)
        self.assertEqual(netlist_utils.extract_number("10 MHz"), 10.0)
        self.assertIsNone(netlist_utils.extract_number("no number"))

    def test_validate_tool_chain_ok(self):
        chain = {
            "tool_calls": [
                {"name": "ac_simulation"},
                {"name": "ac_gain"},
            ]
        }
        toolchain.validate_tool_chain(chain)

    def test_validate_tool_chain_invalid_name(self):
        chain = {"tool_calls": [{"name": "ac_simulation"}, {"name": "unknown_tool"}]}
        with self.assertRaises(ValueError):
            toolchain.validate_tool_chain(chain)

    def test_validate_tool_chain_analysis_before_sim(self):
        chain = {"tool_calls": [{"name": "ac_gain"}, {"name": "ac_simulation"}]}
        with self.assertRaises(ValueError):
            toolchain.validate_tool_chain(chain)


if __name__ == "__main__":
    unittest.main()
