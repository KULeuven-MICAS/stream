"""Canonical model-architecture builders as introspectable affine workload graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from xdsl.dialects.builtin import FixedBitwidthType, bf16
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.data_movement import slice_node
from stream.workload.node import ComputationNode, InEdge, NormalizationNode, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = [
    "AttentionConfig",
    "MambaConfig",
    "GQAConfig",
    "LinearAttentionConfig",
    "KVCacheConfig",
    "build_attention_block",
    "build_mamba_block",
    "build_gqa_block",
    "build_linear_attention_block",
    "build_kv_cache_decode_step",
    "MODEL_CATALOG",
]


def _matmul_maps(rank_batch: int) -> tuple[AffineMap, AffineMap, AffineMap]:
    """Affine maps for a batched projection ``out[b.., m, n] = sum_k A[b.., m, k] W[k, n]``
    (``rank_batch`` leading batch axes on the activation)."""
    b = rank_batch
    dim = AffineExpr.dimension
    batch = tuple(dim(i) for i in range(b))
    pos_m, pos_n, pos_k = b, b + 1, b + 2
    num_dims = b + 3
    a_map = AffineMap(num_dims, 0, (*batch, dim(pos_m), dim(pos_k)))  # A[batch.., m, k]
    w_map = AffineMap(num_dims, 0, (dim(pos_k), dim(pos_n)))  # W[k, n]
    o_map = AffineMap(num_dims, 0, (*batch, dim(pos_m), dim(pos_n)))  # O[batch.., m, n]
    return a_map, w_map, o_map


@dataclass(frozen=True)
class AttentionConfig:
    batch: int = 1
    heads: int = 8
    seq: int = 128
    d_head: int = 64
    dtype: FixedBitwidthType = bf16

    @property
    def d_model(self) -> int:
        return self.heads * self.d_head


def build_attention_block(config: AttentionConfig | None = None) -> Workload:
    """A full multi-head self-attention block on the affine IR: Q/K/V projections, scores,
    schedulable-Softmax, context, output projection."""
    c = config or AttentionConfig()
    b, h, s, dh, dm, dt = c.batch, c.heads, c.seq, c.d_head, c.d_model, c.dtype
    nodes: list = []

    x = Tensor.create("x", dt, (b, s, dm))
    nodes.append(InEdge(name="x", outputs=(x,)))

    # Q/K/V projections, head-split folded in: heads[b,h,s,e] = sum_k X[b,s,k] W[k,h,e]
    proj_maps = (
        AffineMap.from_callable(lambda b_, h_, s_, e, k: (b_, s_, k)),  # X[b,s,k]
        AffineMap.from_callable(lambda b_, h_, s_, e, k: (k, h_, e)),  # W[k,h,e]
        AffineMap.from_callable(lambda b_, h_, s_, e, k: (b_, h_, s_, e)),  # out[b,h,s,e]
    )
    heads: dict[str, Tensor] = {}
    for name in ("q", "k", "v"):
        w = Tensor.create(f"W{name}", dt, (dm, h, dh))
        proj = Tensor.create(name, dt, (b, h, s, dh))
        nodes.append(InEdge(name=f"W{name}", outputs=(w,)))
        nodes.append(
            ComputationNode(
                type="MatMul", name=f"proj_{name}", inputs=(x, w), outputs=(proj,), operand_mapping=proj_maps
            )
        )
        heads[name] = proj

    # scores: S[b,h,i,j] = sum_e Q[b,h,i,e] K[b,h,j,e]  (contract the head dim e = REDUCTION)
    scores = Tensor.create("scores", dt, (b, h, s, s))
    score_maps = (
        AffineMap.from_callable(lambda b_, h_, i, j, e: (b_, h_, i, e)),  # Q
        AffineMap.from_callable(lambda b_, h_, i, j, e: (b_, h_, j, e)),  # K
        AffineMap.from_callable(lambda b_, h_, i, j, e: (b_, h_, i, j)),  # S
    )
    nodes.append(
        ComputationNode(
            type="MatMul", name="scores", inputs=(heads["q"], heads["k"]), outputs=(scores,), operand_mapping=score_maps
        )
    )

    # softmax over the key axis j (position 3): a schedulable NormalizationNode, not a barrier
    probs = Tensor.create("probs", dt, (b, h, s, s))
    identity4d = AffineMap.identity(4)
    nodes.append(
        NormalizationNode(
            type="Softmax",
            name="softmax",
            inputs=(scores,),
            outputs=(probs,),
            operand_mapping=(identity4d, identity4d),
            reduction_axes=(3,),
        )
    )

    # context: O[b,h,i,e] = sum_j P[b,h,i,j] V[b,h,j,e]  (contract key position j = REDUCTION)
    ctx = Tensor.create("context", dt, (b, h, s, dh))
    ctx_maps = (
        AffineMap.from_callable(lambda b_, h_, i, e, j: (b_, h_, i, j)),  # P
        AffineMap.from_callable(lambda b_, h_, i, e, j: (b_, h_, j, e)),  # V
        AffineMap.from_callable(lambda b_, h_, i, e, j: (b_, h_, i, e)),  # O
    )
    nodes.append(
        ComputationNode(
            type="MatMul", name="context", inputs=(probs, heads["v"]), outputs=(ctx,), operand_mapping=ctx_maps
        )
    )

    # output projection, head-merge folded in: Y[b,s,o] = sum_{h,e} O[b,h,s,e] Wo[h,e,o]
    wo = Tensor.create("Wo", dt, (h, dh, dm))
    y = Tensor.create("y", dt, (b, s, dm))
    out_maps = (
        AffineMap.from_callable(lambda b_, s_, o, h_, e: (b_, h_, s_, e)),  # O[b,h,s,e]
        AffineMap.from_callable(lambda b_, s_, o, h_, e: (h_, e, o)),  # Wo[h,e,o]
        AffineMap.from_callable(lambda b_, s_, o, h_, e: (b_, s_, o)),  # Y[b,s,o]
    )
    nodes.append(InEdge(name="Wo", outputs=(wo,)))
    nodes.append(
        ComputationNode(type="MatMul", name="out_proj", inputs=(ctx, wo), outputs=(y,), operand_mapping=out_maps)
    )
    nodes.append(OutEdge(name="y", inputs=(y,)))

    return Workload(nodes)


@dataclass(frozen=True)
class MambaConfig:
    seq: int = 128
    d_model: int = 256
    hidden: int = 512
    dtype: FixedBitwidthType = bf16


def build_mamba_block(config: MambaConfig | None = None) -> Workload:
    """A Mamba-style block: input projection -> selective scan (SEQUENTIAL recurrence,
    chunked-rewritten) -> output projection."""
    c = config or MambaConfig()
    s, dm, hid, dt = c.seq, c.d_model, c.hidden, c.dtype
    nodes: list = []

    x = Tensor.create("x", dt, (s, dm))
    w_in = Tensor.create("W_in", dt, (dm, hid))
    u = Tensor.create("u", dt, (s, hid))
    nodes.append(InEdge(name="x", outputs=(x,)))
    nodes.append(InEdge(name="W_in", outputs=(w_in,)))
    nodes.append(
        ComputationNode(type="MatMul", name="in_proj", inputs=(x, w_in), outputs=(u,), operand_mapping=_matmul_maps(0))
    )

    # selective scan: h[t,d] = h[t-1,d] + u[t,d]  -> t is SEQUENTIAL (state read at t-1)
    h_prev = Tensor.create("h_prev", dt, (s, hid))
    h = Tensor.create("h", dt, (s, hid))
    scan_maps = (
        AffineMap.from_callable(lambda t, d: (t, d)),  # u[t,d]
        AffineMap.from_callable(lambda t, d: (t - 1, d)),  # h_prev[t-1,d] -- the state read
        AffineMap.from_callable(lambda t, d: (t, d)),  # h[t,d] -- the state written
    )
    nodes.append(InEdge(name="h_prev", outputs=(h_prev,)))
    nodes.append(ComputationNode(type="Scan", name="scan", inputs=(u, h_prev), outputs=(h,), operand_mapping=scan_maps))

    w_out = Tensor.create("W_out", dt, (hid, dm))
    y = Tensor.create("y", dt, (s, dm))
    nodes.append(InEdge(name="W_out", outputs=(w_out,)))
    nodes.append(
        ComputationNode(
            type="MatMul", name="out_proj", inputs=(h, w_out), outputs=(y,), operand_mapping=_matmul_maps(0)
        )
    )
    nodes.append(OutEdge(name="y", inputs=(y,)))

    return Workload(nodes)


@dataclass(frozen=True)
class GQAConfig:
    batch: int = 1
    groups: int = 2  # KV groups; MQA is groups=1, MHA is groups=heads
    reps: int = 4  # query heads per group
    seq: int = 64
    d_head: int = 32
    dtype: FixedBitwidthType = bf16


def build_gqa_block(config: GQAConfig | None = None) -> Workload:
    """Grouped-Query Attention core: the head axis is factored into (group ``g``, rep ``r``);
    K/V index ``g`` only, so ``r`` is INVARIANT for them (the KV reuse)."""
    c = config or GQAConfig()
    b, g, r, s, e, dt = c.batch, c.groups, c.reps, c.seq, c.d_head, c.dtype
    q = Tensor.create("q", dt, (b, g, r, s, e))
    k = Tensor.create("k", dt, (b, g, s, e))  # no rep axis -> shared across reps
    v = Tensor.create("v", dt, (b, g, s, e))
    scores = Tensor.create("scores", dt, (b, g, r, s, s))
    probs = Tensor.create("probs", dt, (b, g, r, s, s))
    ctx = Tensor.create("context", dt, (b, g, r, s, e))

    score_maps = (
        AffineMap.from_callable(lambda b_, g_, r_, i, j, e_: (b_, g_, r_, i, e_)),  # Q[b,g,r,i,e]
        AffineMap.from_callable(lambda b_, g_, r_, i, j, e_: (b_, g_, j, e_)),  # K[b,g,j,e] -- no r
        AffineMap.from_callable(lambda b_, g_, r_, i, j, e_: (b_, g_, r_, i, j)),  # S[b,g,r,i,j]
    )
    ctx_maps = (
        AffineMap.from_callable(lambda b_, g_, r_, i, e_, j: (b_, g_, r_, i, j)),  # P[b,g,r,i,j]
        AffineMap.from_callable(lambda b_, g_, r_, i, e_, j: (b_, g_, j, e_)),  # V[b,g,j,e] -- no r
        AffineMap.from_callable(lambda b_, g_, r_, i, e_, j: (b_, g_, r_, i, e_)),  # O[b,g,r,i,e]
    )
    identity5d = AffineMap.identity(5)
    nodes = [
        InEdge(name="q", outputs=(q,)),
        InEdge(name="k", outputs=(k,)),
        InEdge(name="v", outputs=(v,)),
        ComputationNode(type="MatMul", name="scores", inputs=(q, k), outputs=(scores,), operand_mapping=score_maps),
        NormalizationNode(
            type="Softmax",
            name="softmax",
            inputs=(scores,),
            outputs=(probs,),
            operand_mapping=(identity5d, identity5d),
            reduction_axes=(4,),  # the key position j
        ),
        ComputationNode(type="MatMul", name="context", inputs=(probs, v), outputs=(ctx,), operand_mapping=ctx_maps),
        OutEdge(name="context_out", inputs=(ctx,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class LinearAttentionConfig:
    seq: int = 64
    d_k: int = 32
    d_v: int = 32
    dtype: FixedBitwidthType = bf16


def build_linear_attention_block(config: LinearAttentionConfig | None = None) -> Workload:
    """Linear attention in recurrent matrix-state form: ``S_t = S_{t-1} + k_t ⊗ v_t``,
    ``y_t = q_t · S_t``; the ``S[t-1]`` read makes ``t`` SEQUENTIAL."""
    c = config or LinearAttentionConfig()
    s, dk, dv, dt = c.seq, c.d_k, c.d_v, c.dtype
    k = Tensor.create("k", dt, (s, dk))
    v = Tensor.create("v", dt, (s, dv))
    q = Tensor.create("q", dt, (s, dk))
    s_prev = Tensor.create("state_prev", dt, (s, dk, dv))
    state = Tensor.create("state", dt, (s, dk, dv))
    y = Tensor.create("y", dt, (s, dv))

    update_maps = (
        AffineMap.from_callable(lambda t, dk_, dv_: (t, dk_)),  # k[t,dk]
        AffineMap.from_callable(lambda t, dk_, dv_: (t, dv_)),  # v[t,dv]
        AffineMap.from_callable(lambda t, dk_, dv_: (t - 1, dk_, dv_)),  # S[t-1] -- the state carry
        AffineMap.from_callable(lambda t, dk_, dv_: (t, dk_, dv_)),  # S[t]
    )
    out_maps = (
        AffineMap.from_callable(lambda t, dv_, dk_: (t, dk_)),  # q[t,dk]
        AffineMap.from_callable(lambda t, dv_, dk_: (t, dk_, dv_)),  # S[t,dk,dv]
        AffineMap.from_callable(lambda t, dv_, dk_: (t, dv_)),  # y[t,dv]
    )
    nodes = [
        InEdge(name="k", outputs=(k,)),
        InEdge(name="v", outputs=(v,)),
        InEdge(name="q", outputs=(q,)),
        InEdge(name="state_prev", outputs=(s_prev,)),
        ComputationNode(
            type="StateUpdate",
            name="state_update",
            inputs=(k, v, s_prev),
            outputs=(state,),
            operand_mapping=update_maps,
        ),
        ComputationNode(type="MatMul", name="readout", inputs=(q, state), outputs=(y,), operand_mapping=out_maps),
        OutEdge(name="y", inputs=(y,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class KVCacheConfig:
    cache_capacity: int = 128  # allocated buffer length
    valid_len: int = 40  # positions actually written so far (the prefix to attend over)
    d_head: int = 64
    dtype: FixedBitwidthType = bf16


def build_kv_cache_decode_step(config: KVCacheConfig | None = None) -> Workload:
    """One autoregressive decode step: ``Slice`` the valid cache prefix ``cache[0:valid_len]``,
    then the new query attends over it (cache position ``j`` is the reduction)."""
    c = config or KVCacheConfig()
    cap, t, e, dt = c.cache_capacity, c.valid_len, c.d_head, c.dtype
    k_cache = Tensor.create("K_cache", dt, (cap, e))
    v_cache = Tensor.create("V_cache", dt, (cap, e))
    q = Tensor.create("q_new", dt, (1, e))

    k_slice, k_valid = slice_node(k_cache, axis=0, start=0, length=t, name="K_valid")
    v_slice, v_valid = slice_node(v_cache, axis=0, start=0, length=t, name="V_valid")

    scores = Tensor.create("scores", dt, (1, t))
    probs = Tensor.create("probs", dt, (1, t))
    ctx = Tensor.create("context", dt, (1, e))
    score_maps = (
        AffineMap.from_callable(lambda i, j, e_: (i, e_)),  # q[1,e]
        AffineMap.from_callable(lambda i, j, e_: (j, e_)),  # K_valid[j,e]
        AffineMap.from_callable(lambda i, j, e_: (i, j)),  # scores[1,j]
    )
    ctx_maps = (
        AffineMap.from_callable(lambda i, e_, j: (i, j)),  # P[1,j]
        AffineMap.from_callable(lambda i, e_, j: (j, e_)),  # V_valid[j,e]
        AffineMap.from_callable(lambda i, e_, j: (i, e_)),  # context[1,e]
    )
    nodes = [
        InEdge(name="K_cache", outputs=(k_cache,)),
        InEdge(name="V_cache", outputs=(v_cache,)),
        InEdge(name="q_new", outputs=(q,)),
        k_slice,
        v_slice,
        ComputationNode(
            type="MatMul", name="scores", inputs=(q, k_valid), outputs=(scores,), operand_mapping=score_maps
        ),
        NormalizationNode(
            type="Softmax",
            name="softmax",
            inputs=(scores,),
            outputs=(probs,),
            operand_mapping=(AffineMap.identity(2), AffineMap.identity(2)),
            reduction_axes=(1,),  # over the valid cache positions
        ),
        ComputationNode(
            type="MatMul", name="context", inputs=(probs, v_valid), outputs=(ctx,), operand_mapping=ctx_maps
        ),
        OutEdge(name="context_out", inputs=(ctx,)),
    ]
    return Workload(nodes)


@dataclass(frozen=True)
class ModelSpec:
    """A named architecture the model catalog can introspect."""

    key: str
    label: str
    description: str
    build: Callable[[], Workload]


MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="attention",
        label="Multi-Head Attention",
        description=(
            "Q@Kᵀ scores and the P@V context are single affine contractions (REDUCTION), and Softmax "
            "reduces over the key axis but is parallel over batch/head/query, so it fuses along those "
            "axes (flash attention)."
        ),
        build=build_attention_block,
    ),
    ModelSpec(
        key="gqa",
        label="Grouped-Query Attention",
        description=(
            "The same attention, but the head axis is factored into (group, rep) and K/V index the "
            "group only (INVARIANT over rep), so every query head in a group reuses the same K/V — "
            "GQA/MQA's KV saving."
        ),
        build=build_gqa_block,
    ),
    ModelSpec(
        key="linear_attention",
        label="Linear Attention (recurrent)",
        description=(
            "Softmax-free attention as a recurrence over a [d_k, d_v] matrix state; the state read at "
            "t−1 makes the sequence axis SEQUENTIAL (the linear-attention ↔ SSM duality)."
        ),
        build=build_linear_attention_block,
    ),
    ModelSpec(
        key="mamba",
        label="Mamba (SSM recurrence)",
        description=(
            "The selective scan reads its state at t−1 and writes it at t, so the sequence axis is "
            "SEQUENTIAL and the chunked rewrite turns it into a chain of dense per-chunk reductions."
        ),
        build=build_mamba_block,
    ),
    ModelSpec(
        key="kv_cache",
        label="KV-Cache Decode Step",
        description=(
            "One autoregressive decode step: a Slice reads exactly the valid prefix cache[0:t], then "
            "the single new query attends over it, reducing over the cache positions."
        ),
        build=build_kv_cache_decode_step,
    ),
)
