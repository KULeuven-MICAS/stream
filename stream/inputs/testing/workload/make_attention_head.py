"""Build the attention workloads used across the tests: one head, and a multi-head block of any size.

A faithful scaled-dot-product attention head -- one head of a small-transformer block: Q/K/V
projections, scores = Q·Kᵀ, a row-wise Softmax over the key axis, then context = attn·V. The weights
are graph inputs (no stored initializers) so the file stays a few hundred bytes. Run this module to
(re)generate ``attention_head.onnx`` next to it. The Softmax is the point: it parses to a single
NormalizationNode that decomposes to max→exp→sum→div for fusion analysis, reducing over the key axis
and broadcasting over query.

Sizes match the catalog ``AttentionConfig`` (a d_model=512, 8-head, seq=128 block, bf16) -- realistic
modern-transformer numbers, and all clean multiples of 32 so they tile evenly onto a systolic array
(the arbitrary 81/32/fp32 this replaced was neither). This ONNX shows a single head; ``d_model`` is
the full model dimension the head projects from, ``d_head`` the per-head width.
"""

from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper

# One head of a small-transformer block (cf. stream.workload.models.AttentionConfig): 8 heads ×
# d_head=64 = d_model=512, sequence length 128, bf16 activations.
SEQ = 128  # query/key positions (sequence length)
DMODEL = 512  # model dimension the head projects from (heads × d_head)
DHEAD = 64  # per-head dimension
DTYPE = TensorProto.BFLOAT16  # modern activation dtype; the parser maps BFLOAT16 -> bf16


def make_attention_head(path: str | None = None) -> onnx.ModelProto:
    i = helper.make_tensor_value_info("I", DTYPE, [SEQ, DMODEL])
    w_q = helper.make_tensor_value_info("wQ", DTYPE, [DMODEL, DHEAD])
    w_k = helper.make_tensor_value_info("wK", DTYPE, [DMODEL, DHEAD])
    w_v = helper.make_tensor_value_info("wV", DTYPE, [DMODEL, DHEAD])
    y = helper.make_tensor_value_info("Y", DTYPE, [SEQ, DHEAD])

    nodes = [
        helper.make_node("MatMul", ["I", "wQ"], ["Q"]),
        helper.make_node("MatMul", ["I", "wK"], ["K"]),
        helper.make_node("MatMul", ["I", "wV"], ["V"]),
        helper.make_node("Transpose", ["K"], ["Kt"], perm=[1, 0]),
        helper.make_node("MatMul", ["Q", "Kt"], ["scores"]),  # [SEQ, SEQ]
        helper.make_node("Softmax", ["scores"], ["attn"], axis=-1),  # row-wise over keys
        helper.make_node("MatMul", ["attn", "V"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "attention_head", [i, w_q, w_k, w_v], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_head.onnx")
    onnx.save(model, path)
    return model


def make_multi_head_attention(path, seq: int, d_model: int = 4096, heads: int = 32) -> str:
    """Multi-head attention with heads as a batch axis: per-head projections, scores, softmax and context."""
    d_head = d_model // heads
    value = lambda name, shape: helper.make_tensor_value_info(name, DTYPE, shape)  # noqa: E731
    nodes = [helper.make_node("MatMul", ["x", f"w{p}"], [p], name=f"proj_{p}") for p in "QKV"]
    nodes += [
        helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 2, 1], name="transpose_K"),
        helper.make_node("MatMul", ["Q", "Kt"], ["scores"], name="scores"),
        helper.make_node("Softmax", ["scores"], ["attn"], axis=-1, name="softmax"),
        helper.make_node("MatMul", ["attn", "V"], ["ctx"], name="context"),
    ]
    weights = [value(f"w{p}", [heads, d_model, d_head]) for p in "QKV"]
    graph = helper.make_graph(
        nodes, "mha", [value("x", [seq, d_model]), *weights], [value("ctx", [heads, seq, d_head])]
    )
    onnx.save(onnx.shape_inference.infer_shapes(helper.make_model(graph)), path)
    return str(path)


if __name__ == "__main__":
    make_attention_head()
    print("wrote attention_head.onnx")
