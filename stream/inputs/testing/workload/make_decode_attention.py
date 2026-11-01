"""Decode attention over a KV cache with grouped queries, plain and flash-decoding, sized like a Llama-3 70B layer."""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

KV_HEADS = 8
GROUP = 4  # query heads per KV head
DHEAD = 128
DTYPE = TensorProto.BFLOAT16
OPSET = 18


def _save(nodes, name, inputs, outputs, path, initializers=()) -> str:
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)]))
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return str(path)


def _axes(name: str, *axes: int) -> TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(axes)], list(axes))


def make_decode_attention(
    path, batch: int, context: int, kv_heads: int = KV_HEADS, group: int = GROUP, d_head: int = DHEAD, dtype=DTYPE
) -> str:
    value = lambda name, shape: helper.make_tensor_value_info(name, dtype, shape)  # noqa: E731
    nodes = [
        helper.make_node("Einsum", ["q", "K"], ["S"], equation="bgrd,bgsd->bgrs", name="scores"),
        helper.make_node("Softmax", ["S"], ["P"], axis=-1, name="softmax"),
        helper.make_node("Einsum", ["P", "V"], ["O"], equation="bgrs,bgsd->bgrd", name="context"),
    ]
    inputs = [
        value("q", [batch, kv_heads, group, d_head]),
        value("K", [batch, kv_heads, context, d_head]),
        value("V", [batch, kv_heads, context, d_head]),
    ]
    return _save(nodes, "decode_attention", inputs, [value("O", [batch, kv_heads, group, d_head])], path)


def make_flash_decoding(
    path,
    batch: int,
    context: int,
    parts: int = 4,
    kv_heads: int = KV_HEADS,
    group: int = GROUP,
    d_head: int = DHEAD,
    dtype=DTYPE,
) -> str:
    """The cache's context split into ``parts`` along axis 2 of K and V, each part attended to on its own."""
    assert context % parts == 0, f"context {context} does not split into {parts} parts"
    value = lambda name, shape: helper.make_tensor_value_info(name, dtype, shape)  # noqa: E731
    node = helper.make_node
    nodes = [
        node("Einsum", ["q", "K"], ["S"], equation="bgrd,bgpsd->bgprs", name="scores"),
        node("ReduceMax", ["S", "last"], ["m"], keepdims=1, name="part_max"),
        node("Sub", ["S", "m"], ["S_shifted"], name="part_shift"),
        node("Exp", ["S_shifted"], ["E"], name="part_exp"),
        node("ReduceSum", ["E", "last"], ["l"], keepdims=1, name="part_sum"),
        node("Einsum", ["E", "V"], ["o"], equation="bgprs,bgpsd->bgprd", name="part_context"),
        node("ReduceMax", ["m", "part"], ["M"], keepdims=1, name="max"),
        node("Sub", ["m", "M"], ["m_shifted"], name="shift"),
        node("Exp", ["m_shifted"], ["w"], name="weight"),
        node("Mul", ["w", "l"], ["wl"], name="weigh_sum"),
        node("ReduceSum", ["wl", "part"], ["L"], keepdims=0, name="sum"),
        node("Mul", ["w", "o"], ["wo"], name="weigh_context"),
        node("ReduceSum", ["wo", "part"], ["N"], keepdims=0, name="combine"),
        node("Div", ["N", "L"], ["O"], name="normalize"),
    ]
    cache = [batch, kv_heads, parts, context // parts, d_head]
    inputs = [value("q", [batch, kv_heads, group, d_head]), value("K", cache), value("V", cache)]
    outputs = [value("O", [batch, kv_heads, group, d_head])]
    return _save(nodes, "flash_decoding", inputs, outputs, path, [_axes("last", -1), _axes("part", 2)])


if __name__ == "__main__":
    for batch in (1, 16):
        for context in (8192, 32768):
            print(make_decode_attention(f"decode_attention_{batch}_{context}.onnx", batch, context))
            print(make_flash_decoding(f"flash_decoding_{batch}_{context}.onnx", batch, context))
