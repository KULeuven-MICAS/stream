"""Tests for Accelerator.get_ir() and parse_accelerator_ir().

Each test builds a real accelerator from one of the checked-in hardware YAML
files, calls get_ir(), and asserts that the structure of the returned dictionary
is correct and internally consistent.
"""

from pathlib import Path

import pytest
import yaml
from zigzag.utils import open_yaml

from stream.api import parse_accelerator_ir
from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.core_validator import ALLOWED_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------

HARDWARE_CASES = [
    pytest.param(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        {
            "name": "tpu_like_quad_core",
            "offchip_core_id": 6,
            "expected_namespace": "zigzag",
            "expected_num_cores": 7,
        },
        id="examples/tpu_like_quad_core",
    ),
    pytest.param(
        "stream/inputs/testing/hardware/tpu_like_quad_core.yaml",
        {
            "name": "tpu_like_quad_core",
            "offchip_core_id": 6,
            "expected_namespace": "zigzag",
            "expected_num_cores": 7,
        },
        id="testing/tpu_like_quad_core",
    ),
    pytest.param(
        "stream/inputs/aie/hardware/whole_array_strix.yaml",
        {
            "name": "whole_array_strix",
            "offchip_core_id": 0,
            "expected_namespace": "aie2",
            "expected_num_cores": 41,
        },
        id="aie/whole_array_strix",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_accelerator(relative_path: str) -> Accelerator:
    path = REPO_ROOT / relative_path
    data = open_yaml(path)
    validator = AcceleratorValidator(data, str(path))
    assert validator.validate(), f"{path} failed validation: {'; '.join(validator.errors)}"
    factory = AcceleratorFactory(validator.normalized_data)
    return factory.create()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_returns_dict(hardware_path, expected):
    """get_ir() must return a non-empty dict."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    assert isinstance(ir, dict)
    assert ir


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_top_level_keys(hardware_path, expected):
    """Top-level IR must contain all mandatory keys."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    required_keys = {"name", "num_cores", "offchip_core_id", "nb_shared_mem_groups", "cores", "core_connectivity"}
    assert required_keys.issubset(ir.keys()), f"Missing keys: {required_keys - ir.keys()}"


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_name_and_metadata(hardware_path, expected):
    """Accelerator name, offchip id, and core count must match the YAML definition."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    assert ir["name"] == expected["name"]
    assert ir["offchip_core_id"] == expected["offchip_core_id"]
    assert ir["num_cores"] == expected["expected_num_cores"]
    assert ir["nb_shared_mem_groups"] >= 1


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_cores_structure(hardware_path, expected):
    """Every core entry must contain the mandatory fields and a valid core_type."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    cores = ir["cores"]
    assert len(cores) == expected["expected_num_cores"]
    # Keys shared by all cores regardless of backend
    common_core_keys = {
        "id",
        "name",
        "core_type",
        "type",
        "row_id",
        "col_id",
        "utilization",
    }
    # Backend-specific keys
    zigzag_extra_keys = {"operational_array", "memory_hierarchy", "dataflows"}
    aie2_extra_keys = {"memory", "max_object_fifo_depth"}
    for core in cores:
        assert common_core_keys.issubset(core.keys()), (
            f"Core {core.get('id')} is missing common keys: {common_core_keys - core.keys()}"
        )
        ns = core["core_type"].split(".")[0] if "." in core["core_type"] else ""
        if ns == "aie2":
            assert aie2_extra_keys.issubset(core.keys()), (
                f"AIE2 core {core.get('id')} is missing keys: {aie2_extra_keys - core.keys()}"
            )
        else:
            assert zigzag_extra_keys.issubset(core.keys()), (
                f"ZigZag core {core.get('id')} is missing keys: {zigzag_extra_keys - core.keys()}"
            )
        assert core["core_type"].startswith(expected["expected_namespace"] + "."), (
            f"Core {core['id']} has unexpected namespace in core_type '{core['core_type']}'"
        )
        assert core["type"] in ALLOWED_KINDS, f"Core {core['id']} has unexpected type '{core['type']}'"


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_core_ids_are_unique(hardware_path, expected):
    """Core ids must be unique and the count must match num_cores.

    Note: ids are not guaranteed to be sequential – cores that are defined in
    the hardware YAML but have no connectivity edges are absent from the
    CoreGraph and therefore absent from the IR (e.g. isolated shim-DMA tiles
    in whole_array_strix).
    """
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    ids = [core["id"] for core in ir["cores"]]
    assert len(ids) == len(set(ids)), "Core ids are not unique"
    assert len(ids) == ir["num_cores"]


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_operational_array(hardware_path, expected):
    """Every ZigZag-backed core's operational_array must have dimension_sizes and total_unit_count.
    AIE2 cores do not have an operational_array."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    for core in ir["cores"]:
        ns = core["core_type"].split(".")[0] if "." in core["core_type"] else ""
        if ns == "aie2":
            # AIE2 cores must have memory.capacity_bits instead
            assert "memory" in core, f"AIE2 core {core['id']} missing 'memory' key"
            assert "capacity_bits" in core["memory"], f"AIE2 core {core['id']} memory missing 'capacity_bits'"
            assert core["memory"]["capacity_bits"] >= 0
            continue
        oa = core["operational_array"]
        assert "dimension_sizes" in oa, f"Core {core['id']} oa missing 'dimension_sizes'"
        assert "total_unit_count" in oa, f"Core {core['id']} oa missing 'total_unit_count'"
        assert isinstance(oa["dimension_sizes"], dict)
        assert oa["total_unit_count"] >= 0


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_memory_hierarchy(hardware_path, expected):
    """Every ZigZag-backed core must have at least one memory level with required fields.
    AIE2 cores use a flat memory.capacity_bits field instead."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    required_mem_keys = {
        "name",
        "size_bits",
        "read_cost",
        "write_cost",
        "area",
        "latency",
        "operands",
        "level_per_operand",
        "served_dimensions",
        "bandwidths_max",
        "bandwidths_min",
    }
    for core in ir["cores"]:
        ns = core["core_type"].split(".")[0] if "." in core["core_type"] else ""
        if ns == "aie2":
            # AIE2 cores only carry flat memory capacity
            assert "memory" in core, f"AIE2 core {core['id']} missing 'memory'"
            continue
        mem_hierarchy = core["memory_hierarchy"]
        assert len(mem_hierarchy) >= 1, f"Core {core['id']} has no memory levels"
        for level in mem_hierarchy:
            assert required_mem_keys.issubset(level.keys()), (
                f"Core {core['id']} memory level '{level.get('name')}' missing keys: {required_mem_keys - level.keys()}"
            )
            assert level["size_bits"] > 0, f"Core {core['id']} has a zero-size memory level"
            assert isinstance(level["operands"], list) and level["operands"]


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_type_specific_aie2_fields(hardware_path, expected):
    """aie2.* cores must expose max_object_fifo_depth; zigzag.* cores must not."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    namespace = expected["expected_namespace"]
    for core in ir["cores"]:
        if namespace == "aie2":
            assert "max_object_fifo_depth" in core, f"aie2 core {core['id']} is missing 'max_object_fifo_depth'"
            assert core["max_object_fifo_depth"] > 0
        else:
            assert "max_object_fifo_depth" not in core, (
                f"non-aie2 core {core['id']} should not have 'max_object_fifo_depth'"
            )


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_connectivity_structure(hardware_path, expected):
    """Every connectivity entry must be a dict with 'type', 'bandwidth', and core references."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    for entry in ir["core_connectivity"]:
        assert "type" in entry, f"Connectivity entry missing 'type': {entry}"
        assert entry["type"] in {"link", "bus"}, f"Unknown connectivity type: {entry['type']}"
        assert "bandwidth" in entry, f"Connectivity entry missing 'bandwidth': {entry}"
        if entry["type"] == "bus":
            assert "cores" in entry, f"Bus entry missing 'cores': {entry}"
            assert isinstance(entry["cores"], list) and len(entry["cores"]) >= 2  # noqa: PLR2004
        else:
            assert "from_core" in entry, f"Link entry missing 'from_core': {entry}"
            assert "to_core" in entry, f"Link entry missing 'to_core': {entry}"


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_get_ir_connectivity_references_valid_core_ids(hardware_path, expected):
    """Every core id referenced in connectivity must exist in the cores list."""
    accel = _build_accelerator(hardware_path)
    ir = accel.get_ir()
    valid_ids = {core["id"] for core in ir["cores"]}
    for entry in ir["core_connectivity"]:
        if entry["type"] == "bus":
            for cid in entry["cores"]:
                assert cid in valid_ids, f"Bus references unknown core id {cid}"
        else:
            assert entry["from_core"] in valid_ids, f"Link 'from_core' {entry['from_core']} not in cores"
            assert entry["to_core"] in valid_ids, f"Link 'to_core' {entry['to_core']} not in cores"


@pytest.mark.parametrize("hardware_path, expected", HARDWARE_CASES)
def test_parse_accelerator_ir_writes_yaml(hardware_path, expected, tmp_path):
    """parse_accelerator_ir() must write a valid YAML file and return its path."""

    out_path = tmp_path / "accel_ir.yaml"
    returned = parse_accelerator_ir(str(REPO_ROOT / hardware_path), str(out_path))

    assert returned == str(out_path)
    assert out_path.exists()

    with open(out_path) as f:
        data = yaml.safe_load(f)

    assert data["name"] == expected["name"]
    assert data["num_cores"] == expected["expected_num_cores"]
    assert data["offchip_core_id"] == expected["offchip_core_id"]
    assert len(data["cores"]) == expected["expected_num_cores"]
