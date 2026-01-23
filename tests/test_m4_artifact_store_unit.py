import pytest

from eesizer_core.runtime.context import RunContext
from eesizer_core.runtime.artifact_store import ArtifactStore
from eesizer_core.contracts.errors import ValidationError


def test_artifact_store_writes_and_indexes(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    recorder = ctx.recorder()
    store = ArtifactStore(recorder)

    entry = store.put("foo", {"a": 1}, producer="test")
    assert entry["name"] == "foo"
    assert entry["kind"] == "json"
    assert entry["sha256"]

    path = recorder.run_dir / entry["path"]
    assert path.exists()
    assert store.get("foo") == {"a": 1}

    index_path = store.dump_index()
    assert index_path.exists()
    payload = index_path.read_text(encoding="utf-8")
    assert "foo" in payload


def test_artifact_store_blocks_path_traversal(tmp_path):
    ctx = RunContext(workspace_root=tmp_path)
    store = ArtifactStore(ctx.recorder())
    with pytest.raises(ValidationError):
        store.put("../evil", {"x": 1})
