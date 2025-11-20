"""Draft package for consolidating shared agent infrastructure."""

from .analysis import aggregate_measurement_values, standard_measurements
from .config import (
    AgentConfig,
    ConfigLoader,
    OptimizationConfig,
    OrchestratorConfig,
    OutputPathPolicy,
    RunPathLayout,
    SimulationConfig,
    ToolConfig,
)
from .context import ArtifactKind, ArtifactRecord, ContextManager, EnvironmentMetadata, ExecutionContext
from .messaging import Message, MessageBundle, MessageRole, ToolCall, ToolResult
from .netlist import NetlistData, NetlistSummary, load_netlist, summarize_netlist
from .prompts import PromptLibrary
from .simulation import MockNgSpiceSimulator, NgSpiceRunner
from .spice import ControlDeck
from .agents.simple import OptimizationTargets, SimpleSizingAgent
from .toolchain import ToolChainExecutor, ToolChainParser, ToolRegistry

__all__ = [
    "AgentConfig",
    "OrchestratorConfig",
    "ConfigLoader",
    "OptimizationConfig",
    "SimulationConfig",
    "OutputPathPolicy",
    "RunPathLayout",
    "ToolConfig",
    "aggregate_measurement_values",
    "standard_measurements",
    "ArtifactKind",
    "ArtifactRecord",
    "ExecutionContext",
    "ContextManager",
    "EnvironmentMetadata",
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
    "NgSpiceRunner",
    "SimpleSizingAgent",
    "OptimizationTargets",
    "ControlDeck",
    "PromptLibrary",
    "ToolChainExecutor",
    "ToolChainParser",
    "ToolRegistry",
]
