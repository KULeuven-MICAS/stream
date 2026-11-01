"""Attention rewritten into its split-sequence form: found by structure, numerically the attention it replaces, and a
graph that grows by its combine whatever the part count."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from stream.frontends import load_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_decode_attention import make_decode_attention, make_flash_decoding
from stream.workload.node import ComputationNode
from stream.workload.normalization import expand_normalizations
from stream.workload.rewrites import get_workload_rewrite
from stream.workload.utils import affine_coefficients
from stream.workload.workload import Workload

REWRITE = get_workload_rewrite("flash_decoding")
_RNG = np.random.default_rng(20261101)
_REDUCE = {"ReduceMax": np.maximum, "ReduceSum": np.add}


def _evaluate(workload: Workload, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every tensor of ``workload`` computed point by point over each node's iteration space, its inputs viewed in the
    shape the workload declares them in."""
    values = {
        edge.outputs[0].name: inputs[edge.outputs[0].name].reshape(edge.outputs[0].shape)
        for edge in workload.get_in_edges()
    }
    for node in workload.dataflow_sort():
        if not isinstance(node, ComputationNode):
            continue
        grid = np.indices([workload.get_dimension_size(d) for d in workload.get_dims(node)])

        def index(tensor, node=node, grid=grid):
            terms = (affine_coefficients(r, node.num_dims) for r in node.get_mapping(tensor).results)
            return tuple(
                const + sum(c * g for c, g in zip(coefficients, grid, strict=True)) for const, coefficients in terms
            )

        x, *rest = (values[t.name][index(t)] for t in node.inputs)
        if node.type == "Exp":
            point = np.exp(x - rest[0]) if rest else np.exp(x)
        else:
            point = {
                "Add": np.add,
                "Sub": np.subtract,
                "Mul": np.multiply,
                "Einsum": np.multiply,
                "MatMul": np.multiply,
                "Div": np.divide,
                "ReduceMax": lambda a: a,
                "ReduceSum": lambda a: a,
            }[node.type](x, *rest)
        out = node.outputs[0]
        reduce = _REDUCE.get(node.type, np.add)
        result = np.full(out.shape, -np.inf if node.type == "ReduceMax" else 0.0)
        reduce.at(result, tuple(np.broadcast_to(i, point.shape).ravel() for i in index(out)), point.ravel())
        values[out.name] = result
    return values


def _value(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _masked_attention(path, heads: int, queries: int, keys: int, d_head: int) -> str:
    """Attention whose scores are scaled and masked before the softmax."""
    nodes = [
        helper.make_node("Einsum", ["q", "k"], ["s"], equation="hqd,hkd->hqk", name="scores"),
        helper.make_node("Mul", ["s", "scale"], ["scaled"], name="scale"),
        helper.make_node("Add", ["scaled", "mask"], ["masked"], name="mask"),
        helper.make_node("Softmax", ["masked"], ["p"], axis=-1, name="softmax"),
        helper.make_node("Einsum", ["p", "v"], ["o"], equation="hqk,hkd->hqd", name="context"),
    ]
    inputs = [
        _value("q", [heads, queries, d_head]),
        _value("k", [heads, keys, d_head]),
        _value("v", [heads, keys, d_head]),
        _value("scale", [1]),
        _value("mask", [queries, keys]),
    ]
    graph = helper.make_graph(nodes, "masked_attention", inputs, [_value("o", [heads, queries, d_head])])
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))
    onnx.save(model, path)
    return str(path)


def _written_out_attention(path, heads: int, queries: int, keys: int, d_head: int) -> str:
    """Attention with its softmax written out as the ONNX ops it is made of."""
    nodes = [
        helper.make_node("Einsum", ["q", "k"], ["s"], equation="hqd,hkd->hqk", name="scores"),
        helper.make_node("ReduceMax", ["s", "last"], ["m"], keepdims=1, name="max"),
        helper.make_node("Sub", ["s", "m"], ["shifted"], name="shift"),
        helper.make_node("Exp", ["shifted"], ["e"], name="exp"),
        helper.make_node("ReduceSum", ["e", "last"], ["l"], keepdims=1, name="sum"),
        helper.make_node("Div", ["e", "l"], ["p"], name="normalize"),
        helper.make_node("Einsum", ["p", "v"], ["o"], equation="hqk,hkd->hqd", name="context"),
    ]
    inputs = [
        _value("q", [heads, queries, d_head]),
        _value("k", [heads, keys, d_head]),
        _value("v", [heads, keys, d_head]),
    ]
    last = helper.make_tensor("last", TensorProto.INT64, [1], [-1])
    graph = helper.make_graph(nodes, "written_out", inputs, [_value("o", [heads, queries, d_head])], [last])
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))
    onnx.save(model, path)
    return str(path)


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _decode(path):
    return make_decode_attention(path, 1, 32, kv_heads=2, group=3, d_head=4)


def _decode_reference(v):
    return np.einsum("bgrs,bgsd->bgrd", _softmax(np.einsum("bgrd,bgsd->bgrs", v["q"], v["K"])), v["V"])


def _prefill(path):
    return make_multi_head_attention(path, seq=16, d_model=8, heads=2)


def _prefill_reference(v):
    q, k, values = (np.einsum("sm,hmd->hsd", v["x"], v[f"w{p}"]) for p in "QKV")
    return np.einsum("hqk,hkd->hqd", _softmax(np.einsum("hqd,hkd->hqk", q, k)), values)


def _masked(path):
    return _masked_attention(path, heads=2, queries=5, keys=24, d_head=4)


def _masked_reference(v):
    scores = np.einsum("hqd,hkd->hqk", v["q"], v["k"]) * v["scale"] + v["mask"]
    return np.einsum("hqk,hkd->hqd", _softmax(scores), v["v"])


CASES = {
    "decode": (_decode, _decode_reference, "O"),
    "prefill": (_prefill, _prefill_reference, "ctx"),
    "masked": (_masked, _masked_reference, "o"),
    "masked decode": (lambda p: _masked_attention(p, 2, 1, 24, 4), _masked_reference, "o"),
    "written out": (
        lambda p: _written_out_attention(p, 2, 3, 24, 4),
        lambda v: np.einsum("hqk,hkd->hqd", _softmax(np.einsum("hqd,hkd->hqk", v["q"], v["k"])), v["v"]),
        "o",
    ),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("parts", [2, 4, 8])
@pytest.mark.parametrize("expanded", [True, False])
def test_the_split_sequence_form_computes_the_attention_it_replaces(tmp_path, case, parts, expanded):
    make, reference, output = CASES[case]
    workload = load_workload(make(tmp_path / f"{case}.onnx"))
    workload = expand_normalizations(workload) if expanded else workload
    (attention,) = REWRITE.find(workload)
    inputs = {edge.outputs[0].name: _RNG.standard_normal(edge.outputs[0].shape) for edge in workload.get_in_edges()}
    split = REWRITE.apply(workload, attention, parts)
    np.testing.assert_allclose(_evaluate(split, inputs)[output], reference(inputs), rtol=1e-9, atol=1e-12)


def test_the_graph_grows_by_the_combine_whatever_the_part_count(tmp_path):
    workload = expand_normalizations(load_workload(make_decode_attention(tmp_path / "decode.onnx", 1, 256)))
    (attention,) = REWRITE.find(workload)
    four, many = (REWRITE.apply(workload, attention, parts) for parts in (4, 64))
    assert len(four.nodes) == len(many.nodes) == len(workload.nodes) + 6
    assert [n.name for n in four.get_computation_nodes()] == [n.name for n in many.get_computation_nodes()]
    shapes = {edge.name: edge.outputs[0].shape for edge in many.get_in_edges()}
    assert shapes == {"q": (1, 8, 4, 128), "K": (1, 8, 64, 4, 128), "V": (1, 8, 64, 4, 128)}
    assert [n.name for n in many.get_computation_nodes() if "combine" in n.name][-1] == "context_combine"


@pytest.mark.parametrize("model", ["swiglu_1_16_32.onnx", "2conv_1_8_32_32_16_32_3.onnx"])
def test_a_workload_without_attention_is_left_as_it_is(model):
    assert REWRITE.find(expand_normalizations(load_workload(f"stream/inputs/testing/workload/{model}"))) == []


def test_attention_already_split_is_not_found_again(tmp_path):
    assert REWRITE.find(load_workload(make_flash_decoding(tmp_path / "flash.onnx", 1, 256))) == []
    workload = load_workload("stream/inputs/testing/workload/attention_head.onnx")
    (attention,) = REWRITE.find(workload)
    assert REWRITE.find(REWRITE.apply(workload, attention, 2)) == []


def test_a_part_count_that_does_not_split_the_sequence_is_refused(tmp_path):
    workload = load_workload(_decode(tmp_path / "decode.onnx"))
    (attention,) = REWRITE.find(workload)
    with pytest.raises(ValueError, match="context attends over a sequence of 32, which 3 parts do not split evenly"):
        REWRITE.apply(workload, attention, 3)
