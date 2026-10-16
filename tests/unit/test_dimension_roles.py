"""One tensor axis read in two roles stays two dims: attention reads the rows of its input once as queries and once as
keys, and a weight two layers share once per layer. Coupling every reader to that axis would make two loops of one node
one dim, so the axis is held whole and each reader keeps its own loop."""

import onnx
from onnx import TensorProto, helper

from stream.parser.onnx.model import ONNXModelParser

SEQ, DMODEL = 64, 32


def _value(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)


def _save(path, nodes, inputs, output):
    model = helper.make_model(helper.make_graph(nodes, "graph", inputs, [output]))
    onnx.save(onnx.shape_inference.infer_shapes(model), path)
    return str(path)


def _parsed(path):
    parser = ONNXModelParser(path)
    parser.run()
    return parser.workload


def _loops(workload, node):
    return [workload.leading_dim(d)[0] for d in workload.get_dims(node)]


def _distinct_loops(workload):
    return {
        node.name: len(set(_loops(workload, node))) == len(_loops(workload, node))
        for node in workload.get_computation_nodes()
    }


def test_rows_computed_once_and_read_as_queries_and_as_keys_are_two_dims(tmp_path):
    """The scores read the rows of ``h`` through the query projection and through the key projection: two loops."""
    nodes = [
        helper.make_node("Relu", ["x"], ["h"], name="act"),
        helper.make_node("MatMul", ["h", "wq"], ["q"], name="proj_q"),
        helper.make_node("MatMul", ["h", "wk"], ["k"], name="proj_k"),
        helper.make_node("Transpose", ["k"], ["kt"], perm=[1, 0], name="transpose_k"),
        helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
    ]
    inputs = [_value("x", [SEQ, DMODEL]), _value("wq", [DMODEL, DMODEL]), _value("wk", [DMODEL, DMODEL])]
    workload = _parsed(_save(tmp_path / "scores.onnx", nodes, inputs, _value("scores", [SEQ, SEQ])))
    scores = next(n for n in workload.get_computation_nodes() if n.name == "scores")
    queries, keys = _loops(workload, scores)[:2]
    assert queries != keys
    assert all(_distinct_loops(workload).values())


def test_a_weight_two_chained_layers_share_keeps_each_layers_channels_apart(tmp_path):
    """``y = (x @ w) @ w``: the second layer contracts the first one's output channels, so coupling the two readers of
    ``w`` along its rows would make the first layer's input and output channels one dim."""
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="first"),
        helper.make_node("MatMul", ["h", "w"], ["y"], name="second"),
    ]
    inputs = [_value("x", [SEQ, DMODEL]), _value("w", [DMODEL, DMODEL])]
    workload = _parsed(_save(tmp_path / "chain.onnx", nodes, inputs, _value("y", [SEQ, DMODEL])))
    assert all(_distinct_loops(workload).values())
    assert len(workload.unique_dimensions()[0]) == 4  # rows, and the input, middle and output channels
