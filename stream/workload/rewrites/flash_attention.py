"""Attention -> online-softmax (flash) attention as a SEQUENTIAL scan over key blocks.

The full-softmax attention barrier (softmax spans the whole key axis, so it cannot fuse along that
axis) is *lifted* by the online-softmax algorithm: process the keys in blocks, carrying a running max
``m``, denominator ``l`` and output accumulator ``o`` per query -- O(1) state, independent of the key
length. That is a sequential scan over key blocks, so this rewrite mirrors the chunked pattern
(dense per-block reductions + an inter-block state carry). ``reference.online_softmax_attention`` is
the NumPy golden.

Two granularities live here, both exact:

- **Per-key-block** -- the ``AttentionBlock`` node: one fused flash kernel updating the (m, l, o) state
  from one KV tile. This is how a fused-softmax accelerator sees it.
- **Per-sub-operator** -- :func:`decompose_attention_block`: the affine sub-ops a block expands into
  (two matmuls with *distinct* contractions + the online-softmax stats + the rescale/accumulate). This
  is how a matmul-array + vector-unit accelerator sees it, and it is what lets Stream reason about the
  affine (array) vs non-affine (vector) structure -- the opaque block cannot, because ``scores``
  contracts the head dim while ``context`` contracts the key dim.

Reference decomposition (no causal mask, no batching folded in). Additional flash-attention variants
(masks, tuned block sizes) register through the same rewrite/decomposition interface -- no fork.
"""

from __future__ import annotations

from math import ceil

from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.rewrites.base import RewriteParams
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = ["build_attention_chain", "decompose_attention_block", "FlashAttentionRewrite"]

_ATTENTION_BLOCK = "AttentionBlock"


def build_attention_chain(base: str, seq_q: int, seq_k: int, d_head: int, block: int, dtype) -> Workload:
    """The online-softmax chain: ``ceil(seq_k / block)`` per-key-block attention updates.

    Each block node has iteration dims ``(i, d, c)``: query ``i`` and head/value ``d`` are PARALLEL, the
    block-local key ``c`` is the REDUCTION (the dense intra-block MACs). It reads the query, its key/
    value block and the incoming running state ``(o, l, m)``, and writes the outgoing state. Consecutive
    blocks share the state tensors, so the graph carries the inter-block sequential dependency with an
    O(1)-per-query footprint -- the flash property. The block is the coarse (fused-kernel) view;
    :func:`decompose_attention_block` expands it into its affine sub-operators.
    """
    n_blocks = ceil(seq_k / block)
    dim = AffineExpr.dimension
    q_map = AffineMap(3, 0, (dim(0), dim(1)))  # Q[i, d]
    kv_map = AffineMap(3, 0, (dim(2), dim(1)))  # K_block / V_block[c, d]
    o_map = AffineMap(3, 0, (dim(0), dim(1)))  # o[i, d]  (running output accumulator)
    stat_map = AffineMap(3, 0, (dim(0),))  # l[i] / m[i]  (running denominator / max)

    q = Tensor.create(f"{base}_Q", dtype, (seq_q, d_head))
    o_prev = Tensor.create(f"{base}_o_init", dtype, (seq_q, d_head))
    l_prev = Tensor.create(f"{base}_l_init", dtype, (seq_q,))
    m_prev = Tensor.create(f"{base}_m_init", dtype, (seq_q,))
    nodes: list = [
        InEdge(name=f"{base}_Q", outputs=(q,)),
        InEdge(name=f"{base}_o_init", outputs=(o_prev,)),
        InEdge(name=f"{base}_l_init", outputs=(l_prev,)),
        InEdge(name=f"{base}_m_init", outputs=(m_prev,)),
    ]
    for b in range(n_blocks):
        this_block = min(block, seq_k - b * block)
        k_block = Tensor.create(f"{base}_K{b}", dtype, (this_block, d_head))
        v_block = Tensor.create(f"{base}_V{b}", dtype, (this_block, d_head))
        o_out = Tensor.create(f"{base}_o{b}", dtype, (seq_q, d_head))
        l_out = Tensor.create(f"{base}_l{b}", dtype, (seq_q,))
        m_out = Tensor.create(f"{base}_m{b}", dtype, (seq_q,))
        nodes.append(InEdge(name=f"{base}_K{b}", outputs=(k_block,)))
        nodes.append(InEdge(name=f"{base}_V{b}", outputs=(v_block,)))
        nodes.append(
            ComputationNode(
                type=_ATTENTION_BLOCK,
                name=f"{base}_block{b}",
                inputs=(q, k_block, v_block, o_prev, l_prev, m_prev),
                # o is the last output so the derived iterator types see (i, d) PARALLEL, c REDUCTION.
                outputs=(m_out, l_out, o_out),
                operand_mapping=(q_map, kv_map, kv_map, o_map, stat_map, stat_map, stat_map, stat_map, o_map),
            )
        )
        o_prev, l_prev, m_prev = o_out, l_out, m_out
    nodes.append(OutEdge(name=f"{base}_out", inputs=(o_prev,)))
    return Workload(nodes)


def decompose_attention_block(node: ComputationNode) -> Workload:
    """Expand one ``AttentionBlock`` into its affine sub-operators (its dataflow inside).

    The block ``(q, k, v, o_prev, l_prev, m_prev) -> (m, l, o)`` decomposes into:

    - ``scores``  -- MatMul ``S[i,c] = sum_d Q[i,d] K[c,d]``  (contracts the **head** dim ``d``)
    - ``max``     -- ReduceMax over the block key ``c``
    - ``m``       -- running max ``max(m_prev, block_max)``  (the new state)
    - ``probs``   -- ``exp(S - m)``  (non-affine compute, affine access)
    - ``sum``     -- ReduceSum over ``c``  (this block's denominator contribution)
    - ``context`` -- MatMul ``O_b[i,d] = sum_c P[i,c] V[c,d]``  (contracts the **key** dim ``c``)
    - ``rescale`` -- ``exp(m_prev - m)``  (the online correction factor)
    - ``l`` / ``o`` -- ``state * rescale + contribution``  (the accumulate)

    So the two matmuls (the MAC-array work) contract *different* axes -- ``d`` is REDUCTION in ``scores``
    but PARALLEL in ``context`` -- which a single affine node cannot express; and the softmax stats + the
    rescale (the vector-unit work) are explicit. The intermediate/output tensors are reused from the
    block, so the decomposition is a drop-in refinement of the coarse node.
    """
    q, k, v, o_prev, l_prev, m_prev = node.inputs
    m_out, l_out, o_out = node.outputs
    dt = q.operand_type
    seq_q, d_head = q.shape
    block = k.shape[0]
    base = node.name
    dim = AffineExpr.dimension

    scores = Tensor.create(f"{base}_scores", dt, (seq_q, block))
    block_max = Tensor.create(f"{base}_blockmax", dt, (seq_q,))
    probs = Tensor.create(f"{base}_probs", dt, (seq_q, block))
    denom = Tensor.create(f"{base}_denom", dt, (seq_q,))
    context = Tensor.create(f"{base}_context", dt, (seq_q, d_head))
    rescale = Tensor.create(f"{base}_rescale", dt, (seq_q,))

    # scores[i,c] = sum_d q[i,d] k[c,d]  -- iteration (i=0, c=1, d=2), d is the REDUCTION.
    score_maps = (
        AffineMap(3, 0, (dim(0), dim(2))),
        AffineMap(3, 0, (dim(1), dim(2))),
        AffineMap(3, 0, (dim(0), dim(1))),
    )
    # context[i,d] = sum_c probs[i,c] v[c,d]  -- iteration (i=0, d=1, c=2), c is the REDUCTION.
    ctx_maps = (AffineMap(3, 0, (dim(0), dim(2))), AffineMap(3, 0, (dim(2), dim(1))), AffineMap(3, 0, (dim(0), dim(1))))
    row = AffineMap(2, 0, (dim(0), dim(1)))  # [i, c]
    col = AffineMap(2, 0, (dim(0),))  # [i]  (a per-query statistic, broadcast over c)
    id2, id1 = AffineMap.identity(2), AffineMap.identity(1)

    subs = [
        ComputationNode(
            type="MatMul", name=f"{base}_scores", inputs=(q, k), outputs=(scores,), operand_mapping=score_maps
        ),
        ComputationNode(
            type="ReduceMax", name=f"{base}_max", inputs=(scores,), outputs=(block_max,), operand_mapping=(row, col)
        ),
        ComputationNode(
            type="Maximum",
            name=f"{base}_m",
            inputs=(m_prev, block_max),
            outputs=(m_out,),
            operand_mapping=(id1, id1, id1),
        ),
        ComputationNode(
            type="Exp", name=f"{base}_probs", inputs=(scores, m_out), outputs=(probs,), operand_mapping=(row, col, row)
        ),
        ComputationNode(
            type="ReduceSum", name=f"{base}_sum", inputs=(probs,), outputs=(denom,), operand_mapping=(row, col)
        ),
        ComputationNode(
            type="MatMul", name=f"{base}_context", inputs=(probs, v), outputs=(context,), operand_mapping=ctx_maps
        ),
        ComputationNode(
            type="Exp",
            name=f"{base}_rescale",
            inputs=(m_prev, m_out),
            outputs=(rescale,),
            operand_mapping=(id1, id1, id1),
        ),
        ComputationNode(
            type="ScaleAdd",
            name=f"{base}_l",
            inputs=(l_prev, rescale, denom),
            outputs=(l_out,),
            operand_mapping=(id1, id1, id1, id1),
        ),
        ComputationNode(
            type="ScaleAdd",
            name=f"{base}_o",
            inputs=(o_prev, rescale, context),
            outputs=(o_out,),
            operand_mapping=(id2, col, id2, id2),
        ),
    ]
    ins = [InEdge(name=t.name, outputs=(t,)) for t in (q, k, v, o_prev, l_prev, m_prev)]
    outs = [
        OutEdge(name=f"{base}_m_out", inputs=(m_out,)),
        OutEdge(name=f"{base}_l_out", inputs=(l_out,)),
        OutEdge(name=f"{base}_o_out", inputs=(o_out,)),
    ]
    return Workload([*ins, *subs, *outs])


class FlashAttentionRewrite:
    """Rewrite an ``Attention`` op (inputs Q, K, V) into the online-softmax key-block scan."""

    name = "flash_attention"
    source_type = "Attention"

    def matches(self, node: ComputationNode) -> bool:
        return node.type == self.source_type

    def apply(self, node: ComputationNode, params: RewriteParams) -> Workload:
        q, k, _v = node.inputs
        seq_q, d_head = q.shape
        seq_k = k.shape[0]
        return build_attention_chain(node.name, seq_q, seq_k, d_head, params.chunk_size, q.operand_type)
