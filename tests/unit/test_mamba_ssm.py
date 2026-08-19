"""The Mamba selective-scan state-update block: fusion + tiling must match arXiv:2504.17333 Fig. 7."""

from __future__ import annotations

import tempfile

import numpy as np

from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.iterator_type import sequential_dims
from stream.workload.models import MambaConfig, build_mamba_block

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def _parse_accelerator():
    ctx = StageContext.from_kwargs(accelerator=_ACCELERATOR, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, LeafStage], ctx).run()[0].get("accelerator")


def _generator(config: MambaConfig) -> tuple[GenericMappingGenerator, object]:
    workload = build_mamba_block(config)
    gen = GenericMappingGenerator(
        accelerator=_parse_accelerator(), workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    return gen, workload


# Large enough that the [L,D,N] intermediates overflow on-chip, so the fusion tiles the token axis.
_BIG = MambaConfig(seq=256, d_inner=512, d_state=16)


def test_fusion_plan_streams_the_token_axis_with_state_resident():
    """One fused region streams the SEQUENTIAL token axis L, keeping the [D,N] state resident."""
    gen, workload = _generator(_BIG)
    plan = gen.fusion_tiling_plan()
    assert len(plan) == 1, "the whole state-update block fuses into one region"
    group = plan[0]

    assert group["recurrence"] is True, "the SSM streams its recurrence (token) axis, not a parallel one"
    assert group["streamed_axis"]["size"] == _BIG.seq, "the streamed axis is the token axis L"
    assert group["tile"] < _BIG.seq, "L is actually tiled (memory-bound intermediates overflow)"

    # the resident state is [D, N]: both the channel and the state axis are held on-chip
    state_axes = {a["size"] for a in group["resident_axes"] if a["state"]}
    assert state_axes == {_BIG.d_inner, _BIG.d_state}


def test_tensor_tiles_expose_L_streamed_state_and_A_resident():
    """The [L,D,N] activations are sliced along L while the state matrix A[D,N] stays fully resident."""
    gen, _ = _generator(_BIG)
    tensors = {t["name"]: t for t in gen.fusion_tiling_plan()[0]["tensors"]}

    # A is reused across all timesteps -> never streamed, held resident at full [D, N]
    assert tensors["A"]["streamed"] is False
    assert tensors["A"]["tile"] == [_BIG.d_inner, _BIG.d_state]

    # the discretization intermediates are streamed: their token axis shrinks to the tile
    for name in ("dA", "Abar", "dBx", "h"):
        assert tensors[name]["streamed"] is True
        assert tensors[name]["tile"][0] < _BIG.seq
        assert tensors[name]["tile"][1:] == [_BIG.d_inner, _BIG.d_state], "only the token axis is tiled"

    # the carried state is flagged as the resident recurrence carry
    assert tensors["h_prev"]["state"] is True


def test_intra_core_tiling_tiles_the_token_axis():
    """The fused group streams the SEQUENTIAL token axis and keeps the scan state (D, N) resident."""
    gen, workload = _generator(_BIG)
    group = gen.fusion_tiling_plan()[0]
    sub = workload.split_fusion_groups()[0]
    scan = next(n for n in sub.get_computation_nodes() if n.type == "SelectiveScan")
    token_axis = sub.get_dims(scan)[next(iter(sequential_dims(scan)))]

    assert group["streamed_axis"]["name"] == str(token_axis)
    assert group["recurrence"] is True
    assert group["tile"] < _BIG.seq
    resident = {a["name"] for a in group["resident_axes"] if a["state"]}
    assert resident and str(token_axis) not in resident


def test_token_axis_is_never_inter_core_split():
    """The recurrence token axis is streamed temporally (protected), never split across cores."""
    gen, workload = _generator(_BIG)
    sub = workload.split_fusion_groups()[0]
    cns = tuple(sub.get_computation_nodes())
    scan = next(n for n in cns if n.type == "SelectiveScan")
    token_axis = sub.get_dims(scan)[next(iter(sequential_dims(scan)))]
    protected = gen._protected_dims(sub, cns)
    unroll = gen._inter_core_unrolling(sub, cns)
    assert token_axis in protected
    assert token_axis not in unroll


def test_small_ssm_keeps_the_whole_block_resident():
    """When the [L,D,N] tensors fit on-chip, no token-axis tiling is emitted (keep it resident)."""
    gen, workload = _generator(MambaConfig(seq=4, d_inner=4, d_state=2))
    sub = workload.split_fusion_groups()[0]
    assert gen._auto_fusion_tiling(sub, tuple(sub.get_computation_nodes())) == []


# --------------------------------------------------------------------------------------------------- #
# Numerical correctness: the affine sub-op decomposition IS the selective-scan recurrence (no cheating) #
# --------------------------------------------------------------------------------------------------- #


def _selective_scan_via_subops(delta, a_mat, b_mat, c_mat, x, d_skip):
    """The recurrence the block's sub-ops encode, one numpy line per node. Its correspondence to the
    graph is pinned by ``test_block_wiring_is_the_selective_scan_recurrence``; without that assertion
    this is just a parallel implementation and proves nothing about the IR. Shapes: delta,x [L,D];
    A [D,N]; B,C [L,N]; d_skip [D]. Returns y [L,D]."""
    ll, dd = delta.shape
    d_a = delta[:, :, None] * a_mat[None, :, :]  # dA[t,d,n] = delta[t,d] * A[d,n]
    a_bar = np.exp(d_a)  # Abar = exp(dA)
    d_b = delta[:, :, None] * b_mat[:, None, :]  # dB[t,d,n] = delta[t,d] * B[t,n]
    d_bx = d_b * x[:, :, None]  # dBx[t,d,n] = dB[t,d,n] * x[t,d]
    h = np.zeros((dd, a_mat.shape[1]))
    y = np.empty((ll, dd))
    for t in range(ll):
        h = a_bar[t] * h + d_bx[t]  # h_t = Abar_t ⊙ h_{t-1} + dBx_t   (SEQUENTIAL)
        y[t] = (c_mat[t][None, :] * h).sum(axis=1) + d_skip * x[t]  # y'_t = sum_N C_t·h_t, + D⊙x
    return y


def _selective_scan_direct(delta, a_mat, b_mat, c_mat, x, d_skip):
    """An independent, textbook selective scan (per-timestep, no precompute) for cross-checking."""
    ll, dd = delta.shape
    nn = a_mat.shape[1]
    h = np.zeros((dd, nn))
    y = np.empty((ll, dd))
    for t in range(ll):
        a_bar_t = np.exp(delta[t][:, None] * a_mat)  # [D,N]
        b_bar_t = delta[t][:, None] * b_mat[t][None, :]  # [D,N]
        h = a_bar_t * h + b_bar_t * x[t][:, None]
        y[t] = h @ c_mat[t] + d_skip * x[t]
    return y


def test_block_wiring_is_the_selective_scan_recurrence():
    """Pins every sub-op's operands, tying the numerical reference below to the actual graph wiring."""
    wl = build_mamba_block(MambaConfig(seq=12, d_inner=5, d_state=4))
    wiring = {n.name: (n.type, tuple(t.name for t in n.inputs), n.outputs[0].name) for n in wl.get_computation_nodes()}
    assert wiring == {
        "dA": ("Mul", ("delta", "A"), "dA"),
        "Abar": ("Exp", ("dA",), "Abar"),
        "dB": ("Mul", ("delta", "B"), "dB"),
        "dBx": ("Mul", ("dB", "x"), "dBx"),
        "scan": ("SelectiveScan", ("Abar", "h_prev", "dBx"), "h"),
        "readout": ("MatMul", ("C", "h"), "y_ssm"),
        "skip": ("Mul", ("D_skip", "x"), "Dx"),
        "out": ("Add", ("y_ssm", "Dx"), "y"),
    }


def test_subop_decomposition_matches_direct_selective_scan():
    """The decomposition the block encodes computes the genuine Mamba selective scan."""
    rng = np.random.default_rng(0)
    ll, dd, nn = 12, 5, 4
    delta = rng.random((ll, dd)) * 0.1  # small step keeps exp(dA) stable
    a_mat = -rng.random((dd, nn))  # A negative (stable decay), as in Mamba
    b_mat = rng.standard_normal((ll, nn))
    c_mat = rng.standard_normal((ll, nn))
    x = rng.standard_normal((ll, dd))
    d_skip = rng.standard_normal(dd)
    np.testing.assert_allclose(
        _selective_scan_via_subops(delta, a_mat, b_mat, c_mat, x, d_skip),
        _selective_scan_direct(delta, a_mat, b_mat, c_mat, x, d_skip),
        rtol=1e-10,
        atol=1e-12,
    )


def test_affine_subop_shapes_match_the_reference_intermediates():
    """The affine sub-op output tensors carry exactly the [L,D,N] shapes the reference produces."""
    cfg = MambaConfig(seq=12, d_inner=5, d_state=4)
    wl = build_mamba_block(cfg)
    shapes = {t.name: tuple(t.shape) for n in wl.get_computation_nodes() for t in n.tensors}
    assert shapes["dA"] == shapes["Abar"] == shapes["dBx"] == shapes["h"] == (cfg.seq, cfg.d_inner, cfg.d_state)
    assert shapes["A"] == (cfg.d_inner, cfg.d_state)
    assert shapes["y"] == (cfg.seq, cfg.d_inner)
