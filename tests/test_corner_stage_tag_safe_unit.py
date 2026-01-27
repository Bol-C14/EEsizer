import re

from eesizer_core.strategies.corner_search.measurement import stage_tag_for_corner


def test_stage_tag_for_corner_is_stage_safe() -> None:
    tag = stage_tag_for_corner("c1.value_high")
    assert "." not in tag
    assert re.match(r"^[a-z0-9_-]+$", tag)


def test_stage_tag_for_corner_empty_defaults() -> None:
    assert stage_tag_for_corner("") == "corner"

