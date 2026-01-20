from pathlib import Path

import pytest

from eesizer_core.contracts import CircuitSource
from eesizer_core.contracts.enums import SourceKind
from eesizer_core.contracts.errors import ValidationError
from eesizer_core.sim import CircuitSourceToNetlistBundleOperator


def test_source_adapter_rejects_invalid_base_dir():
    src = CircuitSource(
        kind=SourceKind.spice_netlist,
        text="R1 in out 1k\n",
        metadata={"base_dir": 123},
    )
    op = CircuitSourceToNetlistBundleOperator()

    with pytest.raises(ValidationError, match="base_dir"):
        op.run({"circuit_source": src}, ctx=None)


def test_source_adapter_rejects_invalid_include_files_type():
    src = CircuitSource(
        kind=SourceKind.spice_netlist,
        text="R1 in out 1k\n",
        metadata={"include_files": "bad.sp"},
    )
    op = CircuitSourceToNetlistBundleOperator()

    with pytest.raises(ValidationError, match="include_files"):
        op.run({"circuit_source": src}, ctx=None)


def test_source_adapter_rejects_invalid_include_files_entry():
    src = CircuitSource(
        kind=SourceKind.spice_netlist,
        text="R1 in out 1k\n",
        metadata={"include_files": ["ok.sp", 123]},
    )
    op = CircuitSourceToNetlistBundleOperator()

    with pytest.raises(ValidationError, match="include_files entries"):
        op.run({"circuit_source": src}, ctx=None)


def test_source_adapter_rejects_invalid_extra_search_paths_entry():
    src = CircuitSource(
        kind=SourceKind.spice_netlist,
        text="R1 in out 1k\n",
        metadata={"extra_search_paths": [Path("."), 123]},
    )
    op = CircuitSourceToNetlistBundleOperator()

    with pytest.raises(ValidationError, match="extra_search_paths entries"):
        op.run({"circuit_source": src}, ctx=None)


def test_source_adapter_accepts_path_metadata(tmp_path):
    src = CircuitSource(
        kind=SourceKind.spice_netlist,
        text="R1 in out 1k\n",
        metadata={
            "base_dir": tmp_path,
            "include_files": ["a.sp", Path("b.sp")],
            "extra_search_paths": [Path("libs")],
        },
    )
    op = CircuitSourceToNetlistBundleOperator()
    result = op.run({"circuit_source": src}, ctx=None)

    bundle = result.outputs["netlist_bundle"]
    assert bundle.base_dir == Path(tmp_path)
    assert bundle.include_files == (Path("a.sp"), Path("b.sp"))
    assert bundle.extra_search_paths == (Path("libs"),)
