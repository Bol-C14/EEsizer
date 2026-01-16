import json

from eesizer_core.contracts.provenance import RunManifest, stable_hash_json, stable_hash_str


def test_run_manifest_records_hashes(tmp_path):
    spec_payload = {"objectives": [{"metric": "ac_mag", "target": 1.0, "sense": "ge"}]}
    param_payload = {"params": [{"param_id": "m1.w", "lower": 0.0, "upper": 1.0, "frozen": False}]}
    cfg_payload = {"budget": {"max_iterations": 1}, "seed": 123}

    manifest = RunManifest(
        run_id="run-1",
        workspace=tmp_path,
        inputs={
            "netlist_sha256": stable_hash_str("* test\n"),
            "spec_sha256": stable_hash_json(spec_payload),
            "param_space_sha256": stable_hash_json(param_payload),
            "cfg_sha256": stable_hash_json(cfg_payload),
            "signature": "sig",
        },
    )
    out_path = tmp_path / "run_manifest.json"
    manifest.save_json(out_path)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["inputs"]["netlist_sha256"] == manifest.inputs["netlist_sha256"]
    assert loaded["inputs"]["spec_sha256"] == manifest.inputs["spec_sha256"]
    assert loaded["inputs"]["param_space_sha256"] == manifest.inputs["param_space_sha256"]
