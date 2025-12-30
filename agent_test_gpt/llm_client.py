"""LLM client wrappers for chat completions and tool calling."""

from openai import OpenAI

from agent_test_gpt.logging_utils import get_logger
import os

_logger = get_logger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")
DEFAULT_FUNCTION_MODEL = os.getenv("LLM_FUNCTION_MODEL", "gpt-5.2")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))


"""
All LLM calls go through this provider wrapper.
Single endpoint, no builder.

TODO: hardcoded fields like model, api_base_url and api_key for now.

@param prompt: str, user prompt
@return: str, LLM response
"""


def make_chat_completion_request(prompt):
    # The base URL for your local server (note the trailing slash)
    #print("Generated API key:", api_key)

    # Step 2: Configure OpenAI client to use your custom endpoint
    #openai.base_url = api_base_url  # Ensure trailing slash here
    #openai.api_key = api_key
    client = OpenAI()

    # Optional: If you need to pass organization or project headers, you can do so.
    # For example:
    # openai.organization = "YOUR_ORG_ID"
    # openai.default_project = "YOUR_PROJECT_ID"

    # Step 3: Call the Chat Completions API with streaming enabled
    _logger.info("Making ChatCompletion request with streaming enabled...")
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            stream=True,
            max_completion_tokens=20000,
        )
    except Exception as e:
        _logger.error("Error making API call: %s", e, exc_info=True)
        raise

    generated_content = " "
    _logger.debug("Streaming response:")
    for chunk in response:
        # Each 'chunk' is a dict similar to what the OpenAI API returns.
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            _logger.debug(content)
            generated_content += content

    return generated_content


f""" Function calling method for LLM, so far only one ##run_ngspice##

TODO - Hardcoded fields , e.g. model name, tool name etc.

@ param prompt: str, user prompt
@ return: str, LLM response
"""


def make_chat_completion_request_function(prompt):
    # The base URL for your local server (note the trailing slash)
    #print("Generated API key:", api_key)

    # Step 2: Configure OpenAI client to use your custom endpoint
    #openai.base_url = api_base_url  # Ensure trailing slash here
    #openai.api_key = api_key
    client = OpenAI()

    # Optional: If you need to pass organization or project headers, you can do so.
    # For example:
    # openai.organization = "YOUR_ORG_ID"
    # openai.default_project = "YOUR_PROJECT_ID"

    # Step 3: Call the Chat Completions API with streaming enabled
    _logger.info("Making ChatCompletion request with function calling enabled...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "universal_circuit_tool",
                "description": (
                    "A unified tool that enables simulation and analysis of analog and mixed-signal (AMS) circuits. "
                    "It supports multiple functions including DC, AC, and transient simulations, as well as performance "
                    "analysis such as gain, bandwidth, phase margin, power consumption, and more."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "simulation_type": {
                            "type": "string",
                            "enum": ["dc", "ac", "transient"],
                            "description": "Type of simulation to run: DC, AC, or transient."
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": [
                                "ac_gain", "output_swing", "offset", "ICMR", "tran_gain",
                                "bandwidth", "unity_bandwidth", "phase_margin", "power",
                                "thd_input_range", "cmrr_tran"
                            ],
                            "description": "Type of analysis to perform on the simulation results."
                        },
                        "simulation_tool": {
                            "type": "string",
                            "enum": ["run_ngspice"],
                            "description": "Name of the SPICE simulation tool to use (e.g., run_ngspice)."
                        }
                    },
                    "required": []
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=DEFAULT_FUNCTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            stream=False,
            tools=tools,
            tool_choice="required"
        )
    except Exception as e:
        _logger.error("Error making API call: %s", e, exc_info=True)
        raise

    # Step 4: Print the complete JSON response.
    _logger.debug("Non-streaming response: %s", response)

    return response
