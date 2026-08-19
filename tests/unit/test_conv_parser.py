from __future__ import annotations

import tempfile

import onnx
from onnx import TensorProto, helper
from xdsl.ir.affine import AffineBinaryOpExpr

from stream.parser.onnx.model import ONNXModelParser


def _vi(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def test_conv_accepts_asymmetric_2d_padding():
    weight = helper.make_tensor("W", TensorProto.FLOAT, [4, 8, 3, 3], [])
    node = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        name="ConvAsymPad",
        kernel_shape=[3, 3],
        pads=[1, 2, 0, 3],
    )
    graph = helper.make_graph(
        [node],
        "g",
        [_vi("X", (1, 8, 8, 8))],
        [_vi("Y", (1, 4, 7, 11))],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()

    conv = parser.workload.get_computation_nodes()[0]
    input_map = conv.operand_mapping[0]
    input_y = input_map.results[2]
    input_x = input_map.results[3]

    assert conv.name == "ConvAsymPad"
    assert isinstance(input_y, AffineBinaryOpExpr)
    assert isinstance(input_x, AffineBinaryOpExpr)
    assert int(input_y.eval([0, 0, 0, 0, 0, 0, 0], [])) == -1
    assert int(input_x.eval([0, 0, 0, 0, 0, 0, 0], [])) == -2
