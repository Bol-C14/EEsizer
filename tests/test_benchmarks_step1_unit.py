from __future__ import annotations

import json
from pathlib import Path

from eesizer_core.domain.spice.params import infer_param_space_from_ir
from eesizer_core.domain.spice.sanitize_rules import sanitize_spice_netlist
from eesizer_core.operators.netlist import TopologySignatureOperator


BENCH_ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
BENCHES = ("rc", "ota", "opamp3")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bench_dir(name: str) -> Path:
    return BENCH_ROOT / name


def test_benchmarks_have_expected_metadata():
    for bench in BENCHES:
        bench_path = _bench_dir(bench)
        payload = _read_json(bench_path / "bench.json")
        assert payload.get("name") == bench
        assert payload.get("top_netlist")
        assert payload.get("spec")
        assert isinstance(payload.get("nodes"), dict)
        assert isinstance(payload.get("supplies"), dict)
        knobs = payload.get("recommended_knobs")
        assert isinstance(knobs, list)
        assert knobs


def test_benchmark_specs_are_shape_valid():
    for bench in BENCHES:
        bench_path = _bench_dir(bench)
        payload = _read_json(bench_path / "spec.json")
        assert isinstance(payload.get("objectives"), list)
        assert isinstance(payload.get("constraints"), list)
        assert isinstance(payload.get("observables"), list)
        assert isinstance(payload.get("notes"), dict)


def test_benchmark_includes_are_relative():
    for bench in BENCHES:
        bench_path = _bench_dir(bench)
        netlist_path = bench_path / "bench.sp"
        text = netlist_path.read_text(encoding="utf-8")
        result = sanitize_spice_netlist(text)
        for inc in result.includes:
            inc_path = Path(inc)
            assert not inc_path.is_absolute()
            assert ".." not in inc_path.parts


def test_benchmark_signature_and_param_space():
    op = TopologySignatureOperator()
    for bench in BENCHES:
        bench_path = _bench_dir(bench)
        netlist_path = bench_path / "bench.sp"
        text = netlist_path.read_text(encoding="utf-8")
        result = op.run({"netlist_text": text}, ctx=None).outputs
        signature = result["signature"]
        assert signature
        param_space = infer_param_space_from_ir(result["circuit_ir"])
        if bench == "rc":
            assert len(param_space.params) >= 2
        else:
            assert len(param_space.params) > 0


def test_recommended_knobs_exist_in_param_space():
    op = TopologySignatureOperator()
    for bench in BENCHES:
        bench_path = _bench_dir(bench)
        payload = _read_json(bench_path / "bench.json")
        knobs = [k.lower() for k in payload.get("recommended_knobs", [])]
        netlist_path = bench_path / "bench.sp"
        text = netlist_path.read_text(encoding="utf-8")
        result = op.run({"netlist_text": text}, ctx=None).outputs
        param_space = infer_param_space_from_ir(result["circuit_ir"])
        for knob in knobs:
            assert param_space.contains(knob)
