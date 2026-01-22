import pytest

from eesizer_core.contracts.plan import Action
from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.artifact_store import ArtifactStore
from eesizer_core.runtime.plan_executor import PlanExecutor
from eesizer_core.runtime.tool_registry import ToolRegistry


def test_plan_executor_executes_tools_in_order(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    recorder = ctx.recorder()
    store = ArtifactStore(recorder)
    store.put("x", 2)
    store.put("y", 3)

    reg = ToolRegistry()

    def add(inputs, _ctx, params):
        return {"z": inputs["x"] + inputs["y"]}

    def mul(inputs, _ctx, params):
        return {"w": inputs["z"] * int(params.get("k", 2))}

    reg.register("add", add)
    reg.register("mul", mul)
    plan = [
        Action(op="add", inputs=("x", "y"), outputs=("z",)),
        Action(op="mul", inputs=("z",), outputs=("w",), params={"k": 5}),
    ]

    exe = PlanExecutor(reg)
    events = exe.execute(plan, store=store, ctx=None, recorder=recorder)
    assert store.get("z") == 5
    assert store.get("w") == 25
    assert len(events) == 2
    assert (ctx.run_dir() / "orchestrator" / "plan_execution.jsonl").exists()


def test_plan_executor_raises_on_missing_output(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    store = ArtifactStore(ctx.recorder())
    store.put("x", 1)

    reg = ToolRegistry()

    def bad(inputs, _ctx, params):
        return {"not_declared": 123}

    reg.register("bad", bad)
    exe = PlanExecutor(reg)
    with pytest.raises(Exception):
        exe.execute([Action(op="bad", inputs=("x",), outputs=("y",))], store=store, ctx=None)