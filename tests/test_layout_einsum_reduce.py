"""Layout-only operators fold into the nodes reading them, Einsum and the reductions parse to the loops they walk, and
the three ways frameworks write multi-head attention (PyTorch's reshapes and transposes, JAX's einsums, and a
per-head output projection summed over the heads) all become the same dataflow, costed the same."""

import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from stream.api import SolveOptions, evaluate_mapping
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.node import FusionEdge
from stream.workload.workload import ComputationNode

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
SEQ, DMODEL, HEADS = 128, 512, 8
DHEAD = DMODEL // HEADS


def _value(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)


def _save(path, nodes, inputs, output, initializers=()):
    graph = helper.make_graph(nodes, "graph", inputs, [output], list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(onnx.shape_inference.infer_shapes(model), path)
    return str(path)


def _shape(name, values):
    return numpy_helper.from_array(np.array(values, np.int64), name)


def _pytorch_attention(path):
    """Heads split with Reshape and Transpose, merged back with Transpose and Reshape, as a PyTorch export does."""
    nodes = []
    for p in "QKV":
        nodes += [
            helper.make_node("MatMul", ["x", f"w{p}"], [p], name=f"proj_{p}"),
            helper.make_node("Reshape", [p, "heads"], [f"{p}_h"], name=f"split_{p}"),
            helper.make_node("Transpose", [f"{p}_h"], [f"{p}_t"], perm=[1, 2, 0] if p == "K" else [1, 0, 2]),
        ]
    nodes += [
        helper.make_node("MatMul", ["Q_t", "K_t"], ["scores"], name="scores"),
        helper.make_node("Softmax", ["scores"], ["attn"], axis=-1, name="softmax"),
        helper.make_node("MatMul", ["attn", "V_t"], ["ctx"], name="context"),
        helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[1, 0, 2], name="merge_heads"),
        helper.make_node("Reshape", ["ctx_t", "model"], ["ctx_m"], name="merge"),
        helper.make_node("MatMul", ["ctx_m", "wO"], ["y"], name="proj_O"),
    ]
    inputs = [_value("x", [SEQ, DMODEL])] + [_value(f"w{p}", [DMODEL, DMODEL]) for p in "QKVO"]
    shapes = [_shape("heads", [SEQ, HEADS, DHEAD]), _shape("model", [SEQ, DMODEL])]
    return _save(path, nodes, inputs, _value("y", [SEQ, DMODEL]), shapes)


def _jax_attention(path, sum_heads=False):
    """In einsums; with ``sum_heads``, the output projection per head and a ReduceSum adding the heads."""
    nodes = [helper.make_node("Einsum", ["x", f"w{p}"], [p], equation="sd,dhk->hsk", name=f"proj_{p}") for p in "QKV"]
    nodes += [
        helper.make_node("Einsum", ["Q", "K"], ["scores"], equation="hsk,htk->hst", name="scores"),
        helper.make_node("Softmax", ["scores"], ["attn"], axis=-1, name="softmax"),
        helper.make_node("Einsum", ["attn", "V"], ["ctx"], equation="hst,htk->hsk", name="context"),
    ]
    initializers = []
    if sum_heads:
        nodes += [
            helper.make_node("Einsum", ["ctx", "wO"], ["y_heads"], equation="hsk,hkd->hsd", name="proj_O"),
            helper.make_node("ReduceSum", ["y_heads", "axes"], ["y"], keepdims=0, name="sum_heads"),
        ]
        initializers = [_shape("axes", [0])]
    else:
        nodes.append(helper.make_node("Einsum", ["ctx", "wO"], ["y"], equation="hsk,hkd->sd", name="proj_O"))
    inputs = [_value("x", [SEQ, DMODEL])] + [_value(f"w{p}", [DMODEL, HEADS, DHEAD]) for p in "QKV"]
    inputs.append(_value("wO", [HEADS, DHEAD, DMODEL]))
    return _save(path, nodes, inputs, _value("y", [SEQ, DMODEL]), initializers)


def _workload(path):
    ctx = StageContext.from_kwargs(accelerator=TPU_V7, workload_path=path, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0].get("workload")


def _nodes(workload):
    return {n.name: n for n in workload.nodes if isinstance(n, ComputationNode)}


def _maps(node):
    return [str(m) for m in node.operand_mapping]


def test_reshapes_and_transposes_fold_into_the_nodes_reading_them(tmp_path):
    """No layout operator is left: the projections write the heads as an axis of their own, the scores read them
    in place, and the output projection contracts heads and head dimension, every axis indexed by one loop."""
    workload = _workload(_pytorch_attention(tmp_path / "attention.onnx"))
    assert not [n for n in workload.nodes if isinstance(n, FusionEdge)]
    nodes = _nodes(workload)
    assert nodes["proj_Q"].outputs[0].shape == (SEQ, HEADS, DHEAD)
    assert _maps(nodes["scores"]) == [
        "(d0, d1, d2, d3) -> (d1, d0, d3)",
        "(d0, d1, d2, d3) -> (d2, d0, d3)",
        "(d0, d1, d2, d3) -> (d0, d1, d2)",
    ]
    assert _maps(nodes["proj_O"]) == [
        "(d0, d1, d2, d3) -> (d2, d0, d3)",
        "(d0, d1, d2, d3) -> (d2, d3, d1)",
        "(d0, d1, d2, d3) -> (d0, d1)",
    ]


def test_a_sum_over_heads_folds_into_the_projection_producing_them(tmp_path):
    workload = _workload(_jax_attention(tmp_path / "attention.onnx", sum_heads=True))
    nodes = _nodes(workload)
    assert "sum_heads" not in nodes
    assert nodes["proj_O"].outputs[0].shape == (SEQ, DMODEL)
    assert _maps(nodes["proj_O"])[-1] == "(d0, d1, d2, d3) -> (d1, d2)"


@pytest.mark.parametrize("mode", ["fused", "layer_by_layer"])
def test_the_ways_frameworks_write_attention_cost_the_same(tmp_path, mode):
    """Once reshapes, transposes and the sum over heads are folded, the three are one dataflow."""
    cycles = []
    for build in (_pytorch_attention, _jax_attention, lambda p: _jax_attention(p, sum_heads=True)):
        path = build(tmp_path / f"attention_{len(cycles)}.onnx")
        cuts = [n.name for n in _workload(path).get_computation_nodes()] if mode == "layer_by_layer" else None
        stage = {"fusion_cut_points": cuts} if cuts else {}
        options = SolveOptions(artifacts=False, stage_options=stage)
        cycles.append(evaluate_mapping(TPU_V7, path, str(tmp_path / str(len(cycles))), options=options).cycles)
    # The folded sum orders its projection's loops differently, which ZigZag's loop ordering search maps a little
    # differently at this small size
    assert cycles[1] == cycles[0]
    assert cycles[2] == pytest.approx(cycles[0], rel=0.02)


def test_fused_attention_splits_every_node_along_the_heads(tmp_path):
    estimate = evaluate_mapping(
        TPU_V7, _jax_attention(tmp_path / "attention.onnx"), str(tmp_path), options=SolveOptions(artifacts=False)
    )
    mapping, workload = estimate.context.get("mapping"), estimate.context.get("workload")
    heads = {
        n.name: {str(d) for d, _ in mapping.get(n).inter_core_tiling[0]}
        for n in workload.get_computation_nodes()
        if n.name in ("scores", "context") or n.name.startswith("softmax")
    }
    assert len(set(map(frozenset, heads.values()))) == 1, heads


def test_a_reshape_that_splits_and_merges_axes_at_once_crosses_memory(tmp_path):
    """[6, 4] viewed as [4, 6] regroups elements across both axes, which no affine map reads in place; the transpose
    after it still folds into its reader."""
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu_a"),
        helper.make_node("Reshape", ["a", "shape"], ["b"], name="regroup"),
        helper.make_node("Transpose", ["b"], ["c"], perm=[1, 0], name="transpose"),
        helper.make_node("Relu", ["c"], ["y"], name="relu_c"),
    ]
    path = _save(
        tmp_path / "regroup.onnx", nodes, [_value("x", [6, 4])], _value("y", [6, 4]), [_shape("shape", [4, 6])]
    )
    workload = _workload(path)
    edges = [n for n in workload.nodes if isinstance(n, FusionEdge)]
    assert [(e.inputs[0].shape, e.outputs[0].shape) for e in edges] == [((6, 4), (4, 6))]
    relu = _nodes(workload)["relu_c"]
    assert relu.inputs[0].name == "b"
    assert _maps(relu)[0] == "(d0, d1) -> (d1, d0)"
    assert len(workload.split_fusion_groups()) == 2


def test_einsum_reads_its_equation(tmp_path):
    nodes = [
        helper.make_node("Einsum", ["a", "b"], ["c"], equation="bij,bjk->bik", name="batched"),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("Einsum", ["r"], ["d"], equation="bik->bki", name="transpose"),
        helper.make_node("Einsum", ["d"], ["e"], equation="bki->bi", name="sum_k"),
    ]
    inputs = [_value("a", [2, 8, 16]), _value("b", [2, 16, 4])]
    workload = _workload(_save(tmp_path / "einsum.onnx", nodes, inputs, _value("e", [2, 8])))
    found = _nodes(workload)
    assert found["batched"].type == "Einsum"
    assert _maps(found["batched"]) == [
        "(d0, d1, d2, d3) -> (d0, d1, d3)",
        "(d0, d1, d2, d3) -> (d0, d3, d2)",
        "(d0, d1, d2, d3) -> (d0, d1, d2)",
    ]
    assert "transpose" not in found  # it only reorders axes: a layout its reader folds in
    assert found["sum_k"].type == "ReduceSum"
    assert found["sum_k"].inputs[0].name == "r"
    assert _maps(found["sum_k"]) == ["(d0, d1, d2) -> (d0, d1, d2)", "(d0, d1, d2) -> (d0, d1)"]


@pytest.mark.parametrize(
    ("op", "axes", "keepdims", "output", "result"),
    [
        ("ReduceSum", [1], 0, [4, 16], "(d0, d1, d2) -> (d0, d2)"),
        ("ReduceMean", [-1], 1, [4, 8, 1], "(d0, d1, d2) -> (d0, d1, 0)"),
        ("ReduceMax", None, 0, [], "(d0, d1, d2) -> ()"),
    ],
)
def test_a_reduction_reduces_the_loops_its_output_drops(tmp_path, op, axes, keepdims, output, result):
    initializers = [_shape("axes", axes)] if axes is not None else []
    inputs = ["r", "axes"] if axes is not None else ["r"]
    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name="relu"),
        helper.make_node(op, inputs, ["y"], keepdims=keepdims, name="reduce"),
    ]
    path = _save(tmp_path / "reduce.onnx", nodes, [_value("x", [4, 8, 16])], _value("y", output), initializers)
    reduce = _nodes(_workload(path))["reduce"]
    assert reduce.type == op
    assert _maps(reduce) == ["(d0, d1, d2) -> (d0, d1, d2)", result]
