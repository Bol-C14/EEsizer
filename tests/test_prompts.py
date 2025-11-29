from eesizer_core.prompts import PromptLibrary


def test_prompt_library_renders_variables():
    library = PromptLibrary()
    template = library.load("tasks_generation_template")
    rendered = template.render(goal="Verify", netlist_summary="One resistor")
    assert "Verify" in rendered
    assert "One resistor" in rendered


def test_prompt_library_overrides_and_search_paths(tmp_path):
    override_path = tmp_path / "simulation_planning.txt"
    override_path.write_text("Override {agent_name} {tool_blueprint}")

    library = PromptLibrary(
        search_paths=[tmp_path],
        overrides={"task_decomposition": "Inline {goal}"},
    )

    inline_template = library.load("task_decomposition")
    assert inline_template.render(goal="X", netlist_summary="") == "Inline X"

    planning_template = library.load("simulation_planning")
    rendered = planning_template.render(
        agent_name="agent", netlist_summary="", tool_blueprint="{}"
    )
    assert rendered.startswith("Override agent")
