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
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    RecordedProvider,
    build_provider,
)
from .simulation import MockNgSpiceSimulator, NgSpiceRunner
from .spice import ControlDeck
from .agents.claude35 import Claude35Agent
from .agents.gemini30 import Gemini30Agent
from .agents.gpt4o import Gpt4oAgent
from .agents.gpt5 import Gpt5Agent
from .agents.gpt5mini import Gpt5MiniAgent
from .agents.scoring import OptimizationTargets, ScoringPolicy
from .agents.reporting import OptimizationReporter
from .agents.simple import SimpleSizingAgent
from .netlist_patch import ParamChange, apply_param_changes
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
    "LLMProvider",
    "LLMResponse",
    "RecordedProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "build_provider",
    "Claude35Agent",
    "Gemini30Agent",
    "Gpt4oAgent",
    "Gpt5Agent",
    "Gpt5MiniAgent",
    "SimpleSizingAgent",
    "OptimizationReporter",
    "ScoringPolicy",
    "OptimizationTargets",
    "ParamChange",
    "apply_param_changes",
    "ControlDeck",
    "PromptLibrary",
    "ToolChainExecutor",
    "ToolChainParser",
    "ToolRegistry",
]
