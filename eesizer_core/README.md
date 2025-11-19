# EEsizer Core Package Draft

This package sketches the production-ready modules that will eventually host the shared runtime used by all notebook-based agents. The layout mirrors the data flow captured in `agent_test_gpt/agent_gpt_openai_flow.md` and provides placeholders for:

- **Agent interfaces** (`agents/base.py`): lifecycle hooks and metadata for orchestrating reasoning → simulation → optimization pipelines.
- **Messaging utilities** (`messaging.py`): strongly typed primitives for prompts, tool invocations, and measurement payloads.
- **Context management** (`context.py`): standardized execution scopes that own working directories, caches, and simulation handles.
- **Configuration system** (`config.py`): layered settings that combine global defaults, per-process overrides, and notebook-specific parameters.

Each module currently focuses on documentation-friendly docstrings and type definitions so other contributors can begin migrating notebook logic without rewriting from scratch.
