import pytest

from eesizer_core.messaging import Message, MessageBundle, MessageRole, ToolCall


def test_message_round_trip_with_tool_calls():
    message = Message(
        role=MessageRole.USER,
        content="Run sim",
        tool_calls=(ToolCall(name="ngspice.run", arguments={"netlist": "inv"}, call_id="call_1"),),
        tags={"stage": "plan"},
    )
    payload = message.to_dict()
    reconstructed = Message.from_dict(payload)
    assert reconstructed.role == message.role
    assert reconstructed.tool_calls[0].name == "ngspice.run"
    assert reconstructed.tool_calls[0].call_id == "call_1"


def test_bundle_parses_openai_shape():
    bundle = MessageBundle(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content="calling tool",
                tool_calls=(ToolCall(name="optimizer", arguments={"gain": 55}, call_id="c1"),),
            )
        ]
    )
    serialized = bundle.as_dict()
    rebuilt = MessageBundle.from_dicts(serialized)
    rebuilt.validate_tool_schema()
    assert rebuilt.messages[0].tool_calls[0].name == "optimizer"


def test_validate_tool_schema_detects_bad_arguments():
    good_call = ToolCall(name="optimizer", arguments={"gain": 55})
    message = Message(role=MessageRole.ASSISTANT, content="ok", tool_calls=(good_call,))
    bundle = MessageBundle(messages=[message])
    bundle.validate_tool_schema()
    good_call.arguments = "bad"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        bundle.validate_tool_schema()
