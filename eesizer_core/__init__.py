"""Draft package for consolidating shared agent infrastructure."""

from .config import AgentConfig, ConfigLoader, OrchestratorConfig
from .context import ContextManager, ExecutionContext
from .messaging import Message, MessageBundle, MessageRole, ToolCall, ToolResult
from .netlist import NetlistData, NetlistSummary, load_netlist, summarize_netlist
from .simulation import MockNgSpiceSimulator
from .agents.simple import OptimizationTargets, SimpleSizingAgent

__all__ = [
    "AgentConfig",
    "OrchestratorConfig",
    "ConfigLoader",
    "ExecutionContext",
    "ContextManager",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "MessageBundle",
    "NetlistData",
    "NetlistSummary",
    "load_netlist",
    "summarize_netlist",
    "MockNgSpiceSimulator",
    "SimpleSizingAgent",
    "OptimizationTargets",
]
