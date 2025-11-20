"""Prompt template loader used by agents to mirror the notebook flow."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Dict, Mapping, MutableMapping


PROMPT_FILE_MAP: Mapping[str, str] = {
    "tasks_generation_template": "tasks_generation_template.txt",
    "target_value_system_prompt": "target_value_system_prompt.txt",
    "simulation_function_explanation": "simulation_function_explanation.txt",
    "analysing_system_prompt": "analysing_system_prompt.txt",
    "optimization_prompt": "optimization_prompt.txt",
    "sizing_prompt": "sizing_prompt.txt",
}


@dataclass(slots=True)
class PromptTemplate:
    """Lightweight wrapper with convenience rendering utilities."""

    name: str
    text: str
    description: str | None = None

    def render(self, **variables: object) -> str:
        return self.text.format(**variables)


class PromptLibrary:
    """Loads templates from package resources and caches them in memory."""

    def __init__(self) -> None:
        self._cache: MutableMapping[str, PromptTemplate] = {}

    def available_prompts(self) -> Dict[str, str]:
        return dict(PROMPT_FILE_MAP)

    def load(self, name: str) -> PromptTemplate:
        if name in self._cache:
            return self._cache[name]
        filename = PROMPT_FILE_MAP.get(name, f"{name}.txt")
        package = resources.files(__name__)
        path = package / filename
        if not path.is_file():
            raise FileNotFoundError(f"Prompt template '{name}' not found at {filename}")
        text = path.read_text(encoding="utf-8").strip()
        template = PromptTemplate(name=name, text=text)
        self._cache[name] = template
        return template


__all__ = ["PromptLibrary", "PromptTemplate", "PROMPT_FILE_MAP"]
