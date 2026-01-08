import contextlib
import io
import types
import unittest
from unittest.mock import patch

from agent_test_gpt import llm_client


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeChat:
    def __init__(self, response):
        self.completions = FakeCompletions(response)


class FakeOpenAI:
    response = None
    last_instance = None

    def __init__(self):
        self.chat = FakeChat(self.response)
        FakeOpenAI.last_instance = self


class TestLLMClientStreaming(unittest.TestCase):
    def test_make_chat_completion_request_streaming(self):
        chunks = [
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="Hello"))]
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="World"))]
            ),
        ]
        FakeOpenAI.response = chunks

        with patch("agent_test_gpt.llm_client.OpenAI", FakeOpenAI):
            with contextlib.redirect_stdout(io.StringIO()):
                result = llm_client.make_chat_completion_request("Hi")

        self.assertEqual(result, " HelloWorld")
        call = FakeOpenAI.last_instance.chat.completions.calls[0]
        self.assertEqual(call["model"], llm_client.DEFAULT_MODEL)
        self.assertEqual(call["messages"], [{"role": "user", "content": "Hi"}])
        self.assertEqual(call["temperature"], llm_client.DEFAULT_TEMPERATURE)
        self.assertEqual(call["stream"], True)
        self.assertEqual(call["max_completion_tokens"], 20000)


class TestLLMClientFunctionCalling(unittest.TestCase):
    def test_make_chat_completion_request_function(self):
        sentinel = {"ok": True}
        FakeOpenAI.response = sentinel

        with patch("agent_test_gpt.llm_client.OpenAI", FakeOpenAI):
            with contextlib.redirect_stdout(io.StringIO()):
                result = llm_client.make_chat_completion_request_function("Run tools")

        self.assertIs(result, sentinel)
        call = FakeOpenAI.last_instance.chat.completions.calls[0]
        self.assertEqual(call["model"], llm_client.DEFAULT_FUNCTION_MODEL)
        self.assertEqual(call["messages"], [{"role": "user", "content": "Run tools"}])
        self.assertEqual(call["temperature"], llm_client.DEFAULT_TEMPERATURE)
        self.assertEqual(call["stream"], False)
        self.assertEqual(call["tool_choice"], "required")

        tools = call["tools"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "universal_circuit_tool")
        analysis_enum = tools[0]["function"]["parameters"]["properties"]["analysis_type"]["enum"]
        self.assertIn("ac_gain", analysis_enum)
        self.assertIn("cmrr_tran", analysis_enum)


if __name__ == "__main__":
    unittest.main()
