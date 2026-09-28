import pytest

from stream.compiler.kernels.library import KernelLibrary

LIBRARY = {
    "family": {"matmul": {"ops_per_cycle": 151.0}, "vector": {"ops_per_cycle": 16.0}},
    "kernel": {
        "matmul_bf16_bf16": {
            "family": "matmul",
            "dims": [{"name": "k", "divisor": 8}, {"name": "n", "divisor": 16}, {"name": "m", "divisor": 16}],
            "cycles": [
                {"m": 64, "k": 64, "n": 64, "cycles": 1595.0},
                {"m": 32, "k": 32, "n": 64, "cycles": 575.0},
            ],
            "per_op": {"cycles": 1595.0, "ops": 262144},
        },
        "partial_softmax": {
            "family": "vector",
            "dims": [{"name": "n", "fixed": 64}, {"name": "m", "blocks": [32, 64]}],
        },
        "silu_bf16": {
            "family": "vector",
            "dims": [{"name": "n", "runtime": True, "keep_whole": True}, {"name": "m", "runtime": True}],
            "per_op": {"cycles": 2503.0, "ops": 2048},
        },
    },
}


def test_a_library_loads_from_toml_yaml_or_a_mapping(tmp_path):
    toml = tmp_path / "kernels.toml"
    toml.write_text('[family.vector]\nops_per_cycle = 16.0\n[kernel.k]\nfamily = "vector"\ndims = [{ name = "m" }]\n')
    yaml = tmp_path / "kernels.yaml"
    yaml.write_text("family: {vector: {ops_per_cycle: 16.0}}\nkernel: {k: {family: vector, dims: [{name: m}]}}\n")
    for source in (toml, str(yaml), KernelLibrary.from_dict(LIBRARY)):
        assert KernelLibrary.load(source) is not None
    assert KernelLibrary.load(None) is None


def test_a_measured_shape_is_priced_at_its_own_call():
    spec = KernelLibrary.from_dict(LIBRARY).spec("matmul_bf16_bf16")
    assert spec.call_cycles({"m": 32, "k": 32, "n": 64}) == (575.0, 32 * 32 * 64)


def test_an_unmeasured_shape_is_priced_from_the_nearest_call():
    spec = KernelLibrary.from_dict(LIBRARY).spec("matmul_bf16_bf16")
    assert spec.call_cycles({"m": 32, "k": 32, "n": 32}) == (575.0, 32 * 32 * 64)


def test_a_kernel_without_measured_shapes_falls_back_to_its_per_op_rate():
    spec = KernelLibrary.from_dict(LIBRARY).spec("silu_bf16")
    assert spec.call_cycles({"m": 32, "n": 64}) == (2503.0, 2048)


def test_the_declared_sizes_are_enforced():
    spec = KernelLibrary.from_dict(LIBRARY).spec("partial_softmax")
    spec.validate({"m": 32, "n": 64})
    with pytest.raises(ValueError, match="n=32"):
        spec.validate({"m": 32, "n": 32})
    with pytest.raises(ValueError, match="m=48"):
        spec.validate({"m": 48, "n": 64})


def test_a_malformed_entry_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        KernelLibrary.from_dict({"kernel": {"k": {"divisor": {"m": 8}}}})
    with pytest.raises(ValueError, match="family"):
        KernelLibrary.from_dict({"kernel": {"k": {"family": "vector", "dims": [{"name": "m"}]}}})
    with pytest.raises(ValueError, match="unknown kernel family"):
        KernelLibrary.from_dict({"family": {"tensor": {"ops_per_cycle": 1.0}}})
    with pytest.raises(ValueError, match="does not declare"):
        KernelLibrary.from_dict(
            {
                "family": {"vector": {"ops_per_cycle": 1.0}},
                "kernel": {"k": {"family": "vector", "dims": [{"name": "m"}], "cycles": [{"q": 1, "cycles": 1.0}]}},
            }
        )


def test_the_matmul_family_carries_the_mac_tile():
    library = KernelLibrary.from_dict({"family": {"matmul": {"ops_per_cycle": 1.0, "mac": {"m": 4, "k": 8, "n": 8}}}})
    assert library.mac == {"m": 4, "k": 8, "n": 8}
    with pytest.raises(ValueError, match="MAC tile"):
        _ = KernelLibrary.from_dict({"family": {"vector": {"ops_per_cycle": 1.0}}}).mac
