"""Attention rewritten into its split-sequence form: found by structure, numerically the attention it replaces, and a
graph that grows by its combine whatever the part count."""

from __future__ import annotations

from functools import partial

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from stream.frontends import load_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_decode_attention import (
    SOFTMAX,
    attention_reference,
    axes_tensor,
    make_decode_attention,
    make_flash_decoding,
)
from stream.workload.node import ComputationNode
from stream.workload.normalization import expand_normalizations
from stream.workload.rewrites.flash_decoding import FlashDecodingRewrite
from stream.workload.utils import affine_coefficients
from stream.workload.workload import Workload

REWRITE = FlashDecodingRewrite()
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


def _decode(path, group=3, softmax=SOFTMAX):
    return make_decode_attention(
        path, 1, 32, kv_heads=2, group=group, d_head=4, dtype=TensorProto.FLOAT, softmax=softmax
    )


def _masked(path, group):
    """Attention whose scores are scaled and masked before the softmax."""
    nodes = [
        helper.make_node("Mul", ["S", "scale"], ["scaled"], name="scale"),
        helper.make_node("Add", ["scaled", "mask"], ["masked"], name="mask"),
        helper.make_node("Softmax", ["masked"], ["P"], axis=-1, name="softmax"),
    ]
    mask = helper.make_tensor("mask", TensorProto.FLOAT, [group, 32], [0.0] * group * 32)
    return _decode(path, group, (nodes, [helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0]), mask]))


WRITTEN_OUT = (
    [
        helper.make_node("ReduceMax", ["S", "last"], ["m"], keepdims=1, name="max"),
        helper.make_node("Sub", ["S", "m"], ["shifted"], name="shift"),
        helper.make_node("Exp", ["shifted"], ["e"], name="exp"),
        helper.make_node("ReduceSum", ["e", "last"], ["l"], keepdims=1, name="sum"),
        helper.make_node("Div", ["e", "l"], ["P"], name="normalize"),
    ],
    [axes_tensor("last", -1)],
)
"""A softmax written out as the ONNX ops it is made of."""


def _reference(v):
    return attention_reference(v["q"], v["K"], v["V"], v.get("scale", 1.0), v.get("mask", 0.0))


def _prefill(path):
    return make_multi_head_attention(path, seq=16, d_model=8, heads=2)


def _prefill_reference(v):
    return attention_reference(*(np.einsum("sm,hmd->hsd", v["x"], v[f"w{p}"]) for p in "QKV"))


CASES = {
    "decode": (_decode, _reference, "O"),
    "prefill": (_prefill, _prefill_reference, "ctx"),
    "masked": (partial(_masked, group=5), _reference, "O"),
    "masked decode": (partial(_masked, group=1), _reference, "O"),
    "written out": (partial(_decode, softmax=WRITTEN_OUT), _reference, "O"),
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


def _written_out(max_axis=-1, summed="e", divisor="l"):
    """A written-out softmax whose maximum, sum or divisor may be taken along another axis or of another tensor."""
    nodes = [
        helper.make_node("ReduceMax", ["S", "max_axis"], ["m"], keepdims=1, name="max"),
        helper.make_node("Sub", ["S", "m"], ["shifted"], name="shift"),
        helper.make_node("Exp", ["shifted"], ["e"], name="exp"),
        helper.make_node("ReduceSum", [summed, "last"], ["l"], keepdims=1, name="sum"),
        helper.make_node("Div", ["e", divisor], ["P"], name="normalize"),
    ]
    return nodes, [axes_tensor("max_axis", max_axis), axes_tensor("last", -1)]


SCORES = helper.make_node("Einsum", ["q", "K"], ["S"], equation="bgrd,bgsd->bgrs", name="scores")
DECODE_INPUTS = {"q": [1, 2, 3, 4], "K": [1, 2, 32, 4], "V": [1, 2, 32, 4]}


def _near_attention(path, scores=(SCORES,), softmax=SOFTMAX, context="bgrs,bgsd->bgrd", inputs=DECODE_INPUTS, also=()):
    """Decode attention with one part changed; ``also`` are further nodes, their outputs graph outputs."""
    value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)  # noqa: E731
    nodes = [*scores, *softmax[0], helper.make_node("Einsum", ["P", "V"], ["O"], equation=context), *also]
    outputs = [value(name, None) for name in ("O", *(n.output[0] for n in also))]
    graph = helper.make_graph(nodes, "near", [value(n, s) for n, s in inputs.items()], outputs, softmax[1])
    onnx.save(onnx.shape_inference.infer_shapes(helper.make_model(graph)), path)
    return str(path)


NEAR_ATTENTION = {
    "a sum of the scores": {"softmax": _written_out(summed="S")},
    "a maximum along another axis": {"softmax": _written_out(max_axis=-2)},
    "a division by the maximum": {"softmax": _written_out(divisor="m")},
    "a context keeping the sequence": {"context": "bgrs,bgsd->bgrsd"},
    "values without the sequence": {"context": "bgrs,bgtd->bgrd"},
    "a context summing the query rows": {"context": "bgrs,bgsd->bgd"},
    "scores that are an input": {"scores": (), "inputs": {"S": [1, 2, 3, 32], "V": [1, 2, 32, 4]}},
    "scores of a broadcast product": {
        "scores": (helper.make_node("Mul", ["q", "K"], ["S"]),),
        "inputs": {"q": [1, 2, 3, 1], "K": [1, 2, 1, 32], "V": [1, 2, 32, 4]},
    },
    "probabilities read outside it": {"also": (helper.make_node("Exp", ["P"], ["read"]),)},
}


def test_the_near_attention_graph_is_attention(tmp_path):
    assert len(REWRITE.find(load_workload(_near_attention(tmp_path / "near.onnx")))) == 1


@pytest.mark.parametrize("change", NEAR_ATTENTION)
def test_attention_with_one_part_changed_is_not_found(tmp_path, change):
    assert REWRITE.find(load_workload(_near_attention(tmp_path / "near.onnx", **NEAR_ATTENTION[change]))) == []


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
