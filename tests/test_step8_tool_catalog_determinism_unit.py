from __future__ import annotations

from eesizer_core.runtime.session_plan_tools import build_session_plan_registry
from eesizer_core.runtime.tool_catalog import build_tool_catalog


def test_step8_tool_catalog_deterministic() -> None:
    reg1 = build_session_plan_registry()
    reg2 = build_session_plan_registry()
    cat1 = build_tool_catalog(reg1)
    cat2 = build_tool_catalog(reg2)
    assert cat1 == cat2
    assert cat1.get("sha256") == cat2.get("sha256")

