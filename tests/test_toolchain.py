from eesizer_core.context import ContextManager
from eesizer_core.messaging import Message, MessageRole, ToolCall, ToolResult
from eesizer_core.toolchain import ToolChainExecutor, ToolChainParser, ToolRegistry


def test_toolchain_parser_and_executor(tmp_path):
    registry = ToolRegistry()

    def _handler(call: ToolCall, context, state):
        state.setdefault("calls", []).append(call.name)
        return ToolResult(call_id=call.call_id or call.name, content={"name": call.name})

    registry.register("alpha", _handler)
    registry.register("beta", _handler)
    parser = ToolChainParser()
    message = Message(
        role=MessageRole.USER,
        content="""tool plan```json\n[{"name": "alpha"}, {"name": "beta"}]\n```""",
    )
    tool_calls = parser.parse([message])
    assert len(tool_calls) == 2
    with ContextManager(run_id="tc", base_dir=tmp_path, config_name="tc") as ctx:
        result = ToolChainExecutor(registry).run(tool_calls, ctx, state={})
    assert result.state["calls"] == ["alpha", "beta"]
    assert "alpha" in result.summary
