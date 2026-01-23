from eesizer_core.contracts import CircuitSource, CircuitSpec, Objective, ParamDef, ParamSpace
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.strategy import OptimizationBudget, StrategyConfig
from eesizer_core.runtime.artifact_store import ArtifactStore
from eesizer_core.runtime.context import RunContext


def test_artifact_store_roundtrip_rehydrates_types(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    store = ArtifactStore(ctx.recorder())

    source = CircuitSource(kind=SourceKind.spice_netlist, text="R1 in out 1k\n.end\n")
    spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))
    param_space = ParamSpace.build([ParamDef(param_id="r1.value"), ParamDef(param_id="c1.value")])
    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=2), notes={"corner_search": {"levels": 2}})

    store.put("source", source)
    store.put("spec", spec)
    store.put("param_space", param_space)
    store.put("cfg", cfg)
    index_path = store.dump_index()

    reload_store = ArtifactStore(ctx.recorder())
    reload_store.load_index(index_path)

    assert isinstance(reload_store.get("source"), CircuitSource)
    assert isinstance(reload_store.get("spec"), CircuitSpec)
    assert isinstance(reload_store.get("param_space"), ParamSpace)
    assert isinstance(reload_store.get("cfg"), StrategyConfig)
