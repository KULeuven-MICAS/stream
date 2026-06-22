"""Unit tests for the per-accelerator setup (config) cost model."""

from types import SimpleNamespace

import pytest

from stream.hardware.architecture.setup_cost import (
    AccfgSetupCostModel,
    ZeroSetupCostModel,
    build_setup_cost_model,
)


def _node(op_type, n_in, n_out, out_shape=None):
    """A minimal duck-typed stand-in for a ComputationNode (optional output shape)."""
    outputs = [SimpleNamespace(shape=out_shape)] if out_shape is not None else [None] * n_out
    return SimpleNamespace(type=op_type, inputs=[None] * n_in, outputs=outputs)


def test_no_spec_is_zero_model():
    model = build_setup_cost_model(None)
    assert isinstance(model, ZeroSetupCostModel)
    assert model.is_zero
    assert model.cycles(_node("Gemm", 2, 1)) == 0
    assert model.activation_cycles == 0


def test_empty_spec_is_zero_model():
    assert isinstance(build_setup_cost_model({}), ZeroSetupCostModel)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown setup_cost kind"):
        build_setup_cost_model({"kind": "made_up"})


def test_accfg_csr_count_and_cycles():
    spec = {
        "kind": "accfg",
        "cycles_per_csr": 12.0,
        "activation_cycles": 288,
        "base_csrs": 6,
        "csrs_per_operand_streamer": 7,
        "per_operation": {"default": 0, "matmulinteger": 0, "requantized_matmul": 8},
    }
    model = build_setup_cost_model(spec)
    assert isinstance(model, AccfgSetupCostModel)
    assert not model.is_zero
    assert model.activation_cycles == 288  # one-time, charged once per accelerator
    # Plain matmul per-layer config: 2 inputs + 1 output -> 3 streamers; 6 + 7*3 + 0 = 27 CSRs -> 324 cyc.
    assert model.cycles(_node("MatMulInteger", 2, 1)) == 324
    # Requantising matmul adds the SIMD unit config: + 8 CSRs -> (27+8)*12 = 420.
    assert model.cycles(_node("requantized_matmul", 2, 1)) == 420


def test_accfg_multilayer_amortises_activation():
    """A 2-layer workload pays activation once + per-layer twice, not 2*(activation+layer)."""
    spec = {"kind": "accfg", "cycles_per_csr": 12.0, "activation_cycles": 288,
            "base_csrs": 6, "csrs_per_operand_streamer": 7}
    model = build_setup_cost_model(spec)
    per_layer = model.cycles(_node("MatMulInteger", 2, 1))
    one_layer = model.activation_cycles + per_layer
    two_layer = model.activation_cycles + 2 * per_layer  # how the scheduler aggregates
    assert one_layer == 612 and two_layer == 936
    assert two_layer < 2 * one_layer  # amortisation: not a naive doubling


def test_accfg_type_is_normalised():
    """Operation kind matches case-insensitively."""
    spec = {"kind": "accfg", "cycles_per_csr": 1.0, "per_operation": {"gemm": 5}}
    model = build_setup_cost_model(spec)
    # 0 base + 0 per-streamer + 5 (gemm) regardless of operand count.
    assert model.cycles(_node("GEMM", 2, 1)) == 5
    assert model.cycles(_node("gemm", 7, 3)) == 5


def test_accfg_more_operands_costs_more():
    spec = {"kind": "accfg", "cycles_per_csr": 1.0, "csrs_per_operand_streamer": 10}
    model = build_setup_cost_model(spec)
    assert model.cycles(_node("x", 2, 1)) == 30  # 3 streamers
    assert model.cycles(_node("x", 3, 1)) == 40  # 4 streamers


def test_default_kind_is_accfg():
    """A spec without an explicit kind defaults to the accfg model."""
    model = build_setup_cost_model({"cycles_per_csr": 2.0, "base_csrs": 5})
    assert isinstance(model, AccfgSetupCostModel)
    assert model.cycles(_node("x", 0, 0)) == 10  # base only (5 * 2.0)


_RESTREAM = {"kind": "accfg", "cycles_per_csr": 0, "restream_cycles_per_tile": 220, "restream_tile_lanes": 8}


def test_restream_is_zero_for_a_single_output_block():
    """N <= lanes -> one block -> no re-stream cost (the gated regime is unchanged)."""
    model = build_setup_cost_model(_RESTREAM)
    assert model.cycles(_node("gemm", 2, 1, out_shape=(64, 8))) == 0   # N=8 -> 1 block
    assert model.cycles(_node("gemm", 2, 1, out_shape=(8, 8))) == 0


def test_restream_scales_with_output_tile_count():
    """Each extra ceil(N/lanes) block adds restream_cycles_per_tile."""
    model = build_setup_cost_model(_RESTREAM)
    assert model.cycles(_node("gemm", 2, 1, out_shape=(16, 16))) == 220   # N=16 -> 2 blocks -> 1 extra
    assert model.cycles(_node("gemm", 2, 1, out_shape=(32, 32))) == 660   # N=32 -> 4 blocks -> 3 extra
    assert model.cycles(_node("gemm", 2, 1, out_shape=(8, 64))) == 7 * 220  # N=64 -> 8 blocks


def test_restream_disabled_by_default_and_safe_without_shape():
    # No restream params -> never adds (existing accelerators unaffected).
    plain = build_setup_cost_model({"kind": "accfg", "cycles_per_csr": 1.0, "base_csrs": 4})
    assert plain.cycles(_node("gemm", 2, 1, out_shape=(32, 32))) == 4
    # Restream configured but output shape unreadable -> falls back to no extra (never raises).
    model = build_setup_cost_model(_RESTREAM)
    assert model.cycles(_node("gemm", 2, 1)) == 0
