"""Prompt template loader used by agents to mirror the notebook flow."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


PROMPT_FILE_MAP: Mapping[str, str] = {
    "tasks_generation_template": "tasks_generation_template.txt",
    "target_value_system_prompt": "target_value_system_prompt.txt",
    "simulation_function_explanation": "simulation_function_explanation.txt",
    "analysing_system_prompt": "analysing_system_prompt.txt",
    "optimization_prompt": "optimization_prompt.txt",
    "sizing_prompt": "sizing_prompt.txt",
    # Stage 2 notebook-parity prompts
    "task_decomposition": "task_decomposition.txt",
    "target_extraction": "target_extraction.txt",
    "simulation_planning": "simulation_planning.txt",
    "optimization_strategy": "optimization_strategy.txt",
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
    """Loads templates from package resources and caches them in memory.

    A library can be constructed with optional ``search_paths`` (for example,
    repo-local prompt bundles) and inline ``overrides`` for agent-specific
    customization. Overrides take precedence over search paths, which in turn
    override the packaged prompt defaults.
    """

    def __init__(
        self,
        *,
        search_paths: Sequence[str | Path] | None = None,
        overrides: Mapping[str, str | Path] | None = None,
    ) -> None:
        self._cache: MutableMapping[str, PromptTemplate] = {}
        self._search_paths: tuple[Path, ...] = tuple(
            Path(path) for path in (search_paths or ())
        )
        self._overrides: MutableMapping[str, str | Path] = dict(overrides or {})

    def available_prompts(self) -> Dict[str, str]:
        return {
            **{name: str(path) for name, path in self._overrides.items()},
            **dict(PROMPT_FILE_MAP),
        }

    def load(self, name: str) -> PromptTemplate:
        if name in self._cache:
            return self._cache[name]

        if name in self._overrides:
            text = self._resolve_override(name, self._overrides[name])
            template = PromptTemplate(name=name, text=text)
            self._cache[name] = template
            return template

        filename = PROMPT_FILE_MAP.get(name, f"{name}.txt")
        search_candidates: Iterable[Path] = (
            (Path(path) / filename for path in self._search_paths)
        )
        for candidate in search_candidates:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                template = PromptTemplate(name=name, text=text)
                self._cache[name] = template
                return template

        package = resources.files(__name__)
        path = package / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt template '{name}' not found. Expected {filename} or overrides."
            )
        text = path.read_text(encoding="utf-8").strip()
        template = PromptTemplate(name=name, text=text)
        self._cache[name] = template
        return template

    def _resolve_override(self, name: str, value: str | Path) -> str:
        path = Path(value)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        if isinstance(value, Path):
            raise FileNotFoundError(
                f"Prompt override for '{name}' expected file at {value} but none found"
            )
        return str(value)


__all__ = ["PromptLibrary", "PromptTemplate", "PROMPT_FILE_MAP"]
