from pathlib import Path

from zigzag.utils import open_yaml

from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.core_validator import core_kind_from_type

REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate_accelerator(relative_path: str):
    path = REPO_ROOT / relative_path
    data = open_yaml(path)
    validator = AcceleratorValidator(data, accelerator_path=str(path))
    assert validator.validate(), f"{path} failed validation: {'; '.join(validator.errors)}"
    return validator.normalized_data


def _assert_core_namespace(normalized_data: dict, expected_namespace: str):
    cores = list(normalized_data["cores"].values())
    assert cores, "No cores were parsed from the accelerator definition."
    assert all(isinstance(core, dict) for core in cores)
    types = [core["type"] for core in cores]
    assert all(t.startswith(f"{expected_namespace}.") for t in types)
    kinds = {core_kind_from_type(t) for t in types}
    assert kinds.issubset({"compute", "memory"})


def test_zigzag_tpu_like_quad_core_parses():
    normalized = _validate_accelerator("stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    _assert_core_namespace(normalized, "zigzag")


def test_aie_whole_array_parses():
    normalized = _validate_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    _assert_core_namespace(normalized, "aie2")


def test_aie_whole_array_strix_parses():
    normalized = _validate_accelerator("stream/inputs/aie/hardware/whole_array_strix.yaml")
    _assert_core_namespace(normalized, "aie2")
