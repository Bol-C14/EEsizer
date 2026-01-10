from eesizer_core.contracts import (
    CircuitSource, CircuitSpec, Objective,
    ParamDef, ParamSpace,
    PatchOp, Patch,
    SimRequest, SimPlan,
    MetricSpec, MetricValue, MetricsBundle,
    SourceKind, SimKind, PatchOpType,
    StrategyConfig, OptimizationBudget,
)


def test_contracts_import_and_instantiate():
    src = CircuitSource(kind=SourceKind.spice_netlist, text=".title t\n.end\n")
    spec = CircuitSpec(objectives=(Objective(metric="ac_gain_db", target=60.0, tol=0.05),))
    ps = ParamSpace.build([ParamDef(param_id="m1.w", unit="m", lower=1e-7, upper=1e-3)])
    patch = Patch(ops=(PatchOp(param="m1.w", op=PatchOpType.mul, value=1.1, why="gm"),))

    plan = SimPlan(sims=(SimRequest(kind=SimKind.ac),))
    ms = MetricSpec(name="ac_gain_db", unit="dB", sim=SimKind.ac, required_files=("output_ac.dat",))
    mb = MetricsBundle(values={"ac_gain_db": MetricValue(name="ac_gain_db", value=None, unit="dB")})

    cfg = StrategyConfig(budget=OptimizationBudget(max_iterations=3))

    assert src.fingerprint().sha256
    assert ps.contains("m1.w")
    assert patch.fingerprint().sha256
    assert plan.sims[0].kind == SimKind.ac
    assert ms.sim == SimKind.ac
    assert "ac_gain_db" in mb.values
    assert cfg.budget.max_iterations == 3
