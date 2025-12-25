import hashlib
import unittest

from agent_test_gpt import prompts


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestPromptTemplates(unittest.TestCase):
    def test_template_hashes(self):
        expected = {
            "TASK_GENERATION_TEMPLATE": "e9d816182e138b2462fef173641dc4dc41f316734febadd6c7ebda4d2d35501f",
            "TARGET_VALUE_SYSTEM_PROMPT": "4dc7e3c5a277c4d8c3496bcf45f1d8fb881fb69dd4ddae0062ddfe4c89c52be2",
            "TYPE_IDENTIFY_TEMPLATE": "6f268d21d244cd13657250e045660dc1d9fb68b2abc2a723194f317df127561a",
            "NODE_SYSTEM_PROMPT": "26d33814e65f00a1a5c485359631a0d07ba550841b1adc7c66c57d37be5193e1",
            "SIMULATION_FUNCTION_EXPLANATION": "beab98558dca3c8aef19698b02170b36c418fd0bd767e42db40230d20e586fdc",
            "ANALYSING_SYSTEM_PROMPT": "d217b92b0adc866ce55a9c51c5a0c6fd9b47f22aad569a98a444c0a55173800c",
            "OPTIMISING_SYSTEM_PROMPT": "d2c4d8594eb74b6fa4900090c70cc33c3aa35193f08bee0d895ce54e80f72d77",
            "SIZING_SYSTEM_PROMPT": "2c939555905816bba2d4527431bd2e8df5300cfce3e614a8994cff283fd8a6e1",
            "SIZING_OUTPUT_TEMPLATE": "5d362c1cea458fcc2b707341872f6a51b535d85a3b48e5649c79f4d44a0342b3",
        }
        values = {
            "TASK_GENERATION_TEMPLATE": prompts.TASK_GENERATION_TEMPLATE,
            "TARGET_VALUE_SYSTEM_PROMPT": prompts.TARGET_VALUE_SYSTEM_PROMPT,
            "TYPE_IDENTIFY_TEMPLATE": prompts.TYPE_IDENTIFY_TEMPLATE,
            "NODE_SYSTEM_PROMPT": prompts.NODE_SYSTEM_PROMPT,
            "SIMULATION_FUNCTION_EXPLANATION": prompts.SIMULATION_FUNCTION_EXPLANATION,
            "ANALYSING_SYSTEM_PROMPT": prompts.ANALYSING_SYSTEM_PROMPT,
            "OPTIMISING_SYSTEM_PROMPT": prompts.OPTIMISING_SYSTEM_PROMPT,
            "SIZING_SYSTEM_PROMPT": prompts.SIZING_SYSTEM_PROMPT,
            "SIZING_OUTPUT_TEMPLATE": prompts.SIZING_OUTPUT_TEMPLATE,
        }
        for name, template in values.items():
            with self.subTest(name=name):
                self.assertEqual(sha256(template), expected[name])


class TestPromptBuilders(unittest.TestCase):
    def test_builder_hashes(self):
        sample_tasks_question = "Optimize this netlist for gain and bandwidth."
        sample_netlist = ".title Example\nM1 out in 0 0 nmos w=1u l=90n\n.end"
        sample_type_question = "Analyze this netlist and tell me the circuit type."
        sample_node_question = "Tell me the input/output nodes."
        sample_user_request = "Simulate and report ac gain and bandwidth."
        sample_previous_results = "gain=10dB, bw=1kHz"
        sample_sizing_question = "Meet gain 20dB and bw 1MHz"
        sample_type_identified = "Two-stage opamp"
        sample_analysis = "Increase W of M1 to raise gain"
        sample_optimisation = "Increase M1 W from 1u to 2u"

        outputs = {
            "build_tasks_generation_prompt": prompts.build_tasks_generation_prompt(
                sample_tasks_question,
                sample_netlist,
            ),
            "build_target_value_prompt": prompts.build_target_value_prompt(sample_tasks_question),
            "build_type_identify_prompt": prompts.build_type_identify_prompt(
                sample_type_question,
                sample_netlist,
            ),
            "build_node_prompt": prompts.build_node_prompt(sample_node_question, sample_netlist),
            "build_simulation_prompt": prompts.build_simulation_prompt(sample_user_request),
            "build_analysis_prompt": prompts.build_analysis_prompt(
                sample_previous_results,
                sample_sizing_question,
            ),
            "build_optimising_prompt": prompts.build_optimising_prompt(
                sample_type_identified,
                sample_analysis,
                sample_previous_results,
            ),
            "build_sizing_prompt": prompts.build_sizing_prompt(
                sample_sizing_question,
                sample_netlist,
                sample_optimisation,
            ),
        }

        expected = {
            "build_tasks_generation_prompt": "6471566b99f2b5929342669da69e269f442e662ade01306f2e7fa0726aa188f8",
            "build_target_value_prompt": "ac3764d0f6f0a7c764a202e47a48b4cbfcc00c071edeb0b7456f5ab2277c67c5",
            "build_type_identify_prompt": "04bb21b7122d8f6bba5145ccae32852d947d82aa0865deb2643381344dfc898a",
            "build_node_prompt": "dcceba5f4e5a3b1b52f84813939dd3201cd54a6731434bfddb2be9dbdc6bbaf9",
            "build_simulation_prompt": "26f3dd7dfa591a17797bcd07726104239d3b481fe4779ab38887370b9f55f93a",
            "build_analysis_prompt": "a7df8f5029ae93c58bd3876e769795b648090b37cfc612b5e529b869022082c5",
            "build_optimising_prompt": "f68e2bfa688b39a049f7c727aaa42839e0434c40fe83af9db9d13ce74eca465c",
            "build_sizing_prompt": "55d6bee8f90f615a22b84b8ad61b53743e6c3ae77deac50ff4ae1aeb826a0fb6",
        }

        for name, output in outputs.items():
            with self.subTest(name=name):
                self.assertEqual(sha256(output), expected[name])


if __name__ == "__main__":
    unittest.main()
