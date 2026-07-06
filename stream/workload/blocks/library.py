"""Modern reference blocks that exercise every ``AccessRelation`` bucket (plan/11, plan/02).

These complement the attention/Mamba catalog (:mod:`stream.workload.models`) with the other core
components of a modern model, chosen so each demonstrates a different access kind:

- **SwiGLU / MLP** -- affine (A) MatMuls + elementwise (B) Silu/Mul: the gated feed-forward.
- **RMSNorm** -- a reduce-then-broadcast ``NormalizationNode``: the reduced axis is a fusion barrier
  (C, static), parallel over the token axis (the flash-norm view).
- **MoE** -- dense per-expert ``[C, d]`` GEMMs (A) with **data-dependent** dispatch/combine
  permutations (C, ``DataDependentAccess``): capacity ``C`` and the load-balance coefficient are DSE
  sweep / calibration knobs. This is the headline data-dependent component.
- **Chunked SSM** -- reuses the M04 ``chunked_scan`` rewrite: a SEQUENTIAL recurrence decomposed into a
  per-chunk reduction chain, chunk size a DSE lever.

All graphs are built on the affine IR exactly as the ONNX parsers produce, so the whole framework
(tiling, cost, fusion, dedup) consumes them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import FixedBitwidthType, bf16
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.access_relation import AffineAccess, PiecewiseAffineAccess
from stream.workload.node import ComputationNode, InEdge, NormalizationNode, OutEdge
from stream.workload.rewrites import RewriteParams, get_rewrite
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = [
    "SwiGLUConfig",
    "RMSNormConfig",
    "MoEConfig",
    "ChunkedSSMConfig",
    "FlashAttentionConfig",
    "build_swiglu_block",
    "build_rmsnorm_block",
    "build_moe_block",
    "build_chunked_ssm_block",
    "build_flash_attention_block",
    "sparse_attention_key_access",
]

# Affine maps for a 2-D matmul O[m,n] = sum_k A[m,k] W[k,n]; iteration dims (m, n, k).
_MM2D: tuple[AffineMap, AffineMap, AffineMap] = (
    AffineMap.from_callable(lambda m, n, k: (m, k)),
    AffineMap.from_callable(lambda m, n, k: (k, n)),
    AffineMap.from_callable(lambda m, n, k: (m, n)),
)


@dataclass(frozen=True)
class SwiGLUConfig:
    tokens: int = 16
    d_model: int = 64
    d_ff: int = 128
    dtype: FixedBitwidthType = bf16


def build_swiglu_block(config: SwiGLUConfig | None = None) -> Workload:
    """SwiGLU feed-forward: ``y = (silu(x @ W_gate) * (x @ W_up)) @ W_down``.

    Two affine gate/up projections, an elementwise Silu + Mul (bucket-B compute-cost tags with affine
    identity access), and the affine down projection -- the canonical gated MLP block."""
    c = config or SwiGLUConfig()
    t, d, f, dt = c.tokens, c.d_model, c.d_ff, c.dtype
    x = Tensor.create("x", dt, (t, d))
    w_gate = Tensor.create("W_gate", dt, (d, f))
    w_up = Tensor.create("W_up", dt, (d, f))
    w_down = Tensor.create("W_down", dt, (f, d))
    gate = Tensor.create("gate", dt, (t, f))
    up = Tensor.create("up", dt, (t, f))
    act = Tensor.create("silu", dt, (t, f))
    hidden = Tensor.create("hidden", dt, (t, f))
    y = Tensor.create("y", dt, (t, d))
    id2 = AffineMap.identity(2)
    nodes = [
        InEdge(name="x", outputs=(x,)),
        InEdge(name="W_gate", outputs=(w_gate,)),
        InEdge(name="W_up", outputs=(w_up,)),
        InEdge(name="W_down", outputs=(w_down,)),
        ComputationNode(type="MatMul", name="gate_proj", inputs=(x, w_gate), outputs=(gate,), operand_mapping=_MM2D),
        ComputationNode(type="MatMul", name="up_proj", inputs=(x, w_up), outputs=(up,), operand_mapping=_MM2D),
        ComputationNode(type="Silu", name="silu", inputs=(gate,), outputs=(act,), operand_mapping=(id2, id2)),
        ComputationNode(
            type="Mul", name="gate_mul", inputs=(act, up), outputs=(hidden,), operand_mapping=(id2, id2, id2)
        ),
        ComputationNode(type="MatMul", name="down_proj", inputs=(hidden, w_down), outputs=(y,), operand_mapping=_MM2D),
        OutEdge(name="y", inputs=(y,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class RMSNormConfig:
    tokens: int = 16
    d_model: int = 64
    dtype: FixedBitwidthType = bf16


def build_rmsnorm_block(config: RMSNormConfig | None = None) -> Workload:
    """RMSNorm as a reduce-then-broadcast + learned scale: ``y = (x / rms(x)) * gamma``.

    The normalization is a ``NormalizationNode`` reducing over the model axis (a fusion barrier along
    that axis, PARALLEL over the token axis); the ``gamma`` scale is an elementwise Mul (bucket B)."""
    c = config or RMSNormConfig()
    t, d, dt = c.tokens, c.d_model, c.dtype
    x = Tensor.create("x", dt, (t, d))
    gamma = Tensor.create("gamma", dt, (d,))
    normed = Tensor.create("normed", dt, (t, d))
    y = Tensor.create("y", dt, (t, d))
    id2 = AffineMap.identity(2)
    gamma_map = AffineMap(2, 0, (AffineExpr.dimension(1),))  # gamma[d] broadcast over the token axis
    nodes = [
        InEdge(name="x", outputs=(x,)),
        InEdge(name="gamma", outputs=(gamma,)),
        NormalizationNode(
            type="RMSNormalization",
            name="rmsnorm",
            inputs=(x,),
            outputs=(normed,),
            operand_mapping=(id2, id2),
            reduction_axes=(1,),  # reduce over the model dimension
        ),
        ComputationNode(
            type="Mul", name="scale", inputs=(normed, gamma), outputs=(y,), operand_mapping=(id2, gamma_map, id2)
        ),
        OutEdge(name="y", inputs=(y,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class MoEConfig:
    tokens: int = 32
    d_model: int = 64
    d_ff: int = 128
    experts: int = 4
    capacity: int = 16  # C: max tokens routed to one expert -- a DSE sweep variable (graph shape)
    load_balance: float = 1.0  # aux load-balance coefficient -- a calibration knob for the router
    dtype: FixedBitwidthType = bf16


def build_moe_block(config: MoEConfig | None = None) -> Workload:
    """A capacity-``C`` Mixture-of-Experts block with data-dependent dispatch/combine.

    Flow: router (affine ``x @ W_r`` -> per-token expert logits) -> **dispatch** (route each token to
    an expert slot: ``MoEDispatch``, a data-dependent permutation reading ``x``) -> per-expert dense
    GEMMs over the ``[E, C, d]`` batch (affine) -> **combine** (scatter expert outputs back to tokens:
    ``MoECombine``, data-dependent). ``access_for`` reports the dispatch/combine data reads as
    :class:`~stream.workload.access_relation.DataDependentAccess` -- their bounding access is the whole
    token/expert axis (worst case), the true routing lifted into the ``capacity`` (sweep) and
    ``load_balance`` (calibration) parameters, never a blocker."""
    c = config or MoEConfig()
    t, d, f, e, cap, dt = c.tokens, c.d_model, c.d_ff, c.experts, c.capacity, c.dtype
    x = Tensor.create("x", dt, (t, d))
    w_r = Tensor.create("W_router", dt, (d, e))
    w1 = Tensor.create("W_expert_in", dt, (e, d, f))
    w2 = Tensor.create("W_expert_out", dt, (e, f, d))
    logits = Tensor.create("router_logits", dt, (t, e))
    dispatched = Tensor.create("dispatched", dt, (e, cap, d))
    h = Tensor.create("expert_hidden", dt, (e, cap, f))
    act = Tensor.create("expert_act", dt, (e, cap, f))
    expert_out = Tensor.create("expert_out", dt, (e, cap, d))
    y = Tensor.create("y", dt, (t, d))

    router_maps = _MM2D  # logits[t,e] = sum_d x[t,d] W_router[d,e]
    dispatch_maps = (
        AffineMap.from_callable(lambda e_, cc, dd, tt: (tt, dd)),  # x[t,d]   (conservative: any token)
        AffineMap.from_callable(lambda e_, cc, dd, tt: (tt, e_)),  # logits[t,e] (the routing index)
        AffineMap.from_callable(lambda e_, cc, dd, tt: (e_, cc, dd)),  # dispatched[e,c,d]
    )
    expert_in_maps = (
        AffineMap.from_callable(lambda e_, cc, ff, dd: (e_, cc, dd)),  # dispatched[e,c,d]
        AffineMap.from_callable(lambda e_, cc, ff, dd: (e_, dd, ff)),  # W_expert_in[e,d,f]
        AffineMap.from_callable(lambda e_, cc, ff, dd: (e_, cc, ff)),  # h[e,c,f]
    )
    id3 = AffineMap.identity(3)
    expert_out_maps = (
        AffineMap.from_callable(lambda e_, cc, dd, ff: (e_, cc, ff)),  # act[e,c,f]
        AffineMap.from_callable(lambda e_, cc, dd, ff: (e_, ff, dd)),  # W_expert_out[e,f,d]
        AffineMap.from_callable(lambda e_, cc, dd, ff: (e_, cc, dd)),  # expert_out[e,c,d]
    )
    combine_maps = (
        AffineMap.from_callable(lambda tt, dd, e_, cc: (e_, cc, dd)),  # expert_out[e,c,d] (conservative)
        AffineMap.from_callable(lambda tt, dd, e_, cc: (tt, e_)),  # logits[t,e] (the combine index)
        AffineMap.from_callable(lambda tt, dd, e_, cc: (tt, dd)),  # y[t,d]
    )
    nodes = [
        InEdge(name="x", outputs=(x,)),
        InEdge(name="W_router", outputs=(w_r,)),
        InEdge(name="W_expert_in", outputs=(w1,)),
        InEdge(name="W_expert_out", outputs=(w2,)),
        ComputationNode(type="MatMul", name="router", inputs=(x, w_r), outputs=(logits,), operand_mapping=router_maps),
        ComputationNode(
            type="MoEDispatch",
            name="dispatch",
            inputs=(x, logits),
            outputs=(dispatched,),
            operand_mapping=dispatch_maps,
        ),
        ComputationNode(
            type="MatMul", name="expert_in", inputs=(dispatched, w1), outputs=(h,), operand_mapping=expert_in_maps
        ),
        ComputationNode(type="Silu", name="expert_silu", inputs=(h,), outputs=(act,), operand_mapping=(id3, id3)),
        ComputationNode(
            type="MatMul", name="expert_out", inputs=(act, w2), outputs=(expert_out,), operand_mapping=expert_out_maps
        ),
        ComputationNode(
            type="MoECombine", name="combine", inputs=(expert_out, logits), outputs=(y,), operand_mapping=combine_maps
        ),
        OutEdge(name="y", inputs=(y,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class ChunkedSSMConfig:
    seq: int = 64
    hidden: int = 32
    chunk_size: int = 16  # the DSE lever: chain length is ceil(seq / chunk_size)
    dtype: FixedBitwidthType = bf16


def build_chunked_ssm_block(config: ChunkedSSMConfig | None = None) -> Workload:
    """A chunked state-space block: a SEQUENTIAL selective scan decomposed into a per-chunk reduction
    chain by the M04 ``chunked_scan`` rewrite. ``chunk_size`` sets the chain length -- the DSE lever
    that trades intra-chunk parallelism against inter-chunk serialization and state buffering."""
    c = config or ChunkedSSMConfig()
    s, hid, dt = c.seq, c.hidden, c.dtype
    x = Tensor.create("x", dt, (s, hid))
    h_prev = Tensor.create("h_prev", dt, (s, hid))
    h = Tensor.create("h", dt, (s, hid))
    scan_maps = (
        AffineMap.from_callable(lambda tt, dd: (tt, dd)),  # x[t,d]
        AffineMap.from_callable(lambda tt, dd: (tt - 1, dd)),  # h_prev[t-1,d] -- the state carry
        AffineMap.from_callable(lambda tt, dd: (tt, dd)),  # h[t,d]
    )
    scan = ComputationNode(type="Scan", name="ssm_scan", inputs=(x, h_prev), outputs=(h,), operand_mapping=scan_maps)
    return get_rewrite("chunked_scan").apply(scan, RewriteParams(chunk_size=c.chunk_size))


@dataclass(frozen=True)
class FlashAttentionConfig:
    seq_q: int = 64
    seq_k: int = 64
    d_head: int = 32
    block_size: int = 16  # key-block size -- the DSE lever (chain length = ceil(seq_k / block_size))
    dtype: FixedBitwidthType = bf16


def build_flash_attention_block(config: FlashAttentionConfig | None = None) -> Workload:
    """Flash attention as an online-softmax scan over key blocks (reuses the M04 ``flash_attention``
    rewrite). Softmax's full-key reduction -- normally a fusion barrier -- is lifted into a SEQUENTIAL
    chain of dense per-block reductions carrying O(1) running state per query. ``block_size`` is the DSE
    lever. The un-decomposed ``Attention`` op (query ``i``, head ``d`` PARALLEL; key ``j`` REDUCTION) is
    what the rewrite consumes."""
    c = config or FlashAttentionConfig()
    dt = c.dtype
    q = Tensor.create("Q", dt, (c.seq_q, c.d_head))
    k = Tensor.create("K", dt, (c.seq_k, c.d_head))
    v = Tensor.create("V", dt, (c.seq_k, c.d_head))
    out = Tensor.create("O", dt, (c.seq_q, c.d_head))
    attn_maps = (
        AffineMap.from_callable(lambda i, j, d: (i, d)),  # Q[i,d]
        AffineMap.from_callable(lambda i, j, d: (j, d)),  # K[j,d]
        AffineMap.from_callable(lambda i, j, d: (j, d)),  # V[j,d]
        AffineMap.from_callable(lambda i, j, d: (i, d)),  # O[i,d]  (j = key is the REDUCTION)
    )
    attention = ComputationNode(
        type="Attention", name="attention", inputs=(q, k, v), outputs=(out,), operand_mapping=attn_maps
    )
    return get_rewrite("flash_attention").apply(attention, RewriteParams(chunk_size=c.block_size))


def sparse_attention_key_access(window: int, dilation: int) -> PiecewiseAffineAccess:
    """The key access of a sparse (local + dilated) attention, as a union of two affine bands.

    A query at position ``i`` attends to a contiguous local band **and** a dilated band of keys (the
    Longformer/BigBird sparse-attention pattern). That receptive field is not one affine map, but each
    band is affine, so the access is a :class:`~stream.workload.access_relation.PiecewiseAffineAccess`
    -- the natural modern producer of the piecewise bucket. Iteration dims are ``(i, w)`` (query
    position, within-band offset); ``footprint`` returns the hull of the two bands. This is the seam a
    frontend / overlay uses to model structured sparsity; ``access_for`` returns the conservative affine
    bound by default."""
    i, w = AffineExpr.dimension(0), AffineExpr.dimension(1)
    local = AffineMap(2, 0, (i - window + w,))  # K[i - window + w]  (contiguous local band)
    dilated = AffineMap(2, 0, (i - dilation * w,))  # K[i - dilation*w]  (dilated band)
    return PiecewiseAffineAccess((AffineAccess(local), AffineAccess(dilated)))
