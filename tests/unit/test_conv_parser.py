from __future__ import annotations

import tempfile

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from torch.nn.functional import conv2d, pad
from xdsl.ir.affine import AffineBinaryOpExpr

from stream.parser.onnx.model import ONNXModelParser
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.workload.affine_transform import AffineTransform


def _vi(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def test_conv_accepts_asymmetric_2d_padding():
    weight = helper.make_tensor("W", TensorProto.FLOAT, [4, 8, 3, 3], [0.0] * (4 * 8 * 3 * 3))
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


CONVS = {
    "strides_per_axis": ((1, 4, 7, 9), (6, 4, 3, 2), {"strides": [2, 1], "pads": [1, 0, 1, 2]}, (0, 2, 1, 1)),
    "dilations_per_axis": (
        (1, 4, 9, 8),
        (6, 4, 3, 3),
        {"strides": [1, 2], "dilations": [2, 1], "pads": [2, 1, 1, 0]},
        (1, 0, 2, 1),
    ),
    "same_upper": ((1, 4, 8, 9), (6, 4, 3, 3), {"strides": [2, 2], "auto_pad": "SAME_UPPER"}, (1, 1, 0, 1)),
    "same_lower": ((1, 4, 8, 9), (6, 4, 3, 3), {"strides": [2, 2], "auto_pad": "SAME_LOWER"}, (1, 1, 1, 0)),
    "valid": ((1, 4, 8, 9), (6, 4, 3, 3), {"strides": [1, 2], "auto_pad": "VALID"}, (0, 0, 0, 0)),
    "grouped": ((1, 4, 7, 7), (6, 2, 3, 3), {"group": 2, "pads": [1, 1, 1, 1]}, (1, 1, 1, 1)),
    "depthwise": ((1, 4, 7, 7), (8, 1, 3, 3), {"group": 4, "strides": [2, 1], "pads": [1, 1, 1, 1]}, (1, 1, 1, 1)),
}


def _parse_conv(shapes: dict[str, tuple[int, ...]], attrs: dict) -> ONNXModelParser:
    node = helper.make_node("Conv", list(shapes), ["Y"], name="Conv", kernel_shape=list(shapes["W"][2:]), **attrs)
    inputs = [_vi(name, shape) for name, shape in shapes.items()]
    graph = helper.make_graph([node], "g", inputs, [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser


@pytest.mark.parametrize(("x_shape", "w_shape", "attrs", "torch_pads"), CONVS.values(), ids=CONVS)
def test_conv_access_maps_compute_torch_conv2d(x_shape, w_shape, attrs, torch_pads):
    """Every point of the parsed iteration space accumulates input times weight into the output its maps name, and
    adds the bias it names once per output: that is torch's conv2d (``torch_pads`` is left, right, top, bottom)."""
    shapes = {"X": x_shape, "W": w_shape, "B": (w_shape[0],)}
    parser = _parse_conv(shapes, attrs)
    (conv,) = parser.workload.get_computation_nodes()
    x, w, b = (np.random.default_rng(seed).standard_normal(shape) for seed, shape in enumerate(shapes.values()))
    sizes = [parser.workload.get_dimension_size(d) for d in parser.workload.get_dims(conv)]
    points = np.indices(sizes).reshape(len(sizes), -1).T
    index = [points @ t.A.T + t.b for t in map(AffineTransform.from_affine_map, conv.operand_mapping)]
    inside = ((index[0] >= 0) & (index[0] < x.shape)).all(1)
    found, bias = np.zeros(conv.outputs[0].shape), np.zeros(conv.outputs[0].shape)
    np.add.at(found, tuple(index[3][inside].T), x[tuple(index[0][inside].T)] * w[tuple(index[1][inside].T)])
    bias[tuple(index[3].T)] = b[tuple(index[2].T)]
    expected = conv2d(
        pad(torch.from_numpy(x), torch_pads),
        torch.from_numpy(w),
        torch.from_numpy(b),
        stride=attrs.get("strides", 1),
        dilation=attrs.get("dilations", 1),
        groups=attrs.get("group", 1),
    )
    np.testing.assert_allclose(found + bias, expected.numpy(), atol=1e-9)


def test_zigzag_prices_a_biased_conv_as_its_product():
    parser = _parse_conv({"X": (1, 4, 7, 7), "W": (6, 4, 3, 3), "B": (6,)}, {"pads": [1, 1, 1, 1]})
    (conv,) = parser.workload.get_computation_nodes()
    estimator = ZigZagCostEstimator(workload=parser.workload, accelerator=None, mapping=None)  # type: ignore[arg-type]
    equation = estimator.create_equation_and_dimension_relations_and_padding_and_pr_sizes(conv)[0]
    assert len(conv.inputs) == 3
    assert [str(op) for op in equation.get_contained_operands()] == ["O", "A", "B"]
