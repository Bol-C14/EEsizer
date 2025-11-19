from eesizer_core.prompts import PromptLibrary


def test_prompt_library_renders_variables():
    library = PromptLibrary()
    template = library.load("tasks_generation_template")
    rendered = template.render(goal="Verify", netlist_summary="One resistor")
    assert "Verify" in rendered
    assert "One resistor" in rendered
