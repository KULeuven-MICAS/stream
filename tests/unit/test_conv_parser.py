from __future__ import annotations

import tempfile

import numpy as np
import onnx
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, helper
from xdsl.ir.affine import AffineBinaryOpExpr

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND
from stream.mapping.mapping import Mapping
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.stages.parsing.accelerator_parser import parse_accelerator
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


def _pair(value) -> tuple[int, int]:
    return tuple(value) if isinstance(value, list) else (value, value)


def _windows(x, kernel, strides, dilations, pads, fill: float = 0.0):
    """The window each output of an (N, C, H, W) input reads, as (N, C, OH, OW, KH, KW); ``pads`` per spatial axis."""
    (dh, dw), (sh, sw) = _pair(dilations), _pair(strides)
    span = [(k - 1) * d + 1 for k, d in zip(kernel, (dh, dw), strict=True)]
    padded = np.pad(x, ((0, 0), (0, 0), *pads), constant_values=fill)
    return sliding_window_view(padded, span, axis=(2, 3))[:, :, ::sh, ::sw, ::dh, ::dw]


def _conv2d(x, w, b, pads, strides=1, dilations=1, group=1):
    """Direct grouped 2-D convolution with bias."""
    win = _windows(x, w.shape[2:], strides, dilations, pads)
    n, c, oh, ow = win.shape[:4]
    win = win.reshape(n, group, c // group, oh, ow, *w.shape[2:])
    out = np.einsum("ngchwij,gkcij->ngkhw", win, w.reshape(group, -1, *w.shape[1:]))
    return out.reshape(n, -1, oh, ow) + b[:, None, None]


def _parse_conv(shapes: dict[str, tuple[int, ...]], attrs: dict) -> ONNXModelParser:
    return _parse("Conv", shapes, {"kernel_shape": list(shapes["W"][2:]), **attrs})


def _parse(op: str, shapes: dict[str, tuple[int, ...]], attrs: dict) -> ONNXModelParser:
    node = helper.make_node(op, list(shapes), ["Y"], name=op, **attrs)
    inputs = [_vi(name, shape) for name, shape in shapes.items()]
    graph = helper.make_graph([node], "g", inputs, [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser


@pytest.mark.parametrize(("x_shape", "w_shape", "attrs", "pads"), CONVS.values(), ids=CONVS)
def test_conv_access_maps_compute_a_direct_conv2d(x_shape, w_shape, attrs, pads):
    """Every point of the parsed iteration space accumulates input times weight into the output its maps name, and
    adds the bias it names once per output: that is a direct conv2d (``pads`` is left, right, top, bottom)."""
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
    left, right, top, bottom = pads
    expected = _conv2d(
        x,
        w,
        b,
        ((top, bottom), (left, right)),
        attrs.get("strides", 1),
        attrs.get("dilations", 1),
        attrs.get("group", 1),
    )
    np.testing.assert_allclose(found + bias, expected, atol=1e-9)


def test_zigzag_prices_a_biased_conv_as_its_product():
    parser = _parse_conv({"X": (1, 4, 7, 7), "W": (6, 4, 3, 3), "B": (6,)}, {"pads": [1, 1, 1, 1]})
    (conv,) = parser.workload.get_computation_nodes()
    estimator = ZigZagCostEstimator(workload=parser.workload, accelerator=None, mapping=None)  # type: ignore[arg-type]
    equation = estimator.create_equation_and_dimension_relations_and_pr_sizes(conv)[0]
    assert len(conv.inputs) == 3
    assert [str(op) for op in equation.get_contained_operands()] == ["O", "A", "B"]


@pytest.mark.parametrize("name", ["grouped", "depthwise"])
def test_zigzag_reads_a_grouped_channel_as_its_group_and_the_channel_within_it(name):
    """A channel ``g*K + k`` is one group and one channel of it, not a window sliding over the groups: only the two
    spatial windows relate dims."""
    x_shape, w_shape, attrs, _ = CONVS[name]
    parser = _parse_conv({"X": x_shape, "W": w_shape}, attrs)
    (conv,) = parser.workload.get_computation_nodes()
    estimator = ZigZagCostEstimator(workload=parser.workload, accelerator=None, mapping=None)  # type: ignore[arg-type]
    relations = estimator.create_equation_and_dimension_relations_and_pr_sizes(conv)[1]
    assert len(relations) == 2


@pytest.mark.parametrize(("kernel", "relations"), [(2, 0), (3, 2)])
def test_zigzag_reads_a_window_no_wider_than_its_stride_as_two_axes(kernel, relations):
    """A 2x2 window of stride 2 reads each input once, the pair of an output and a kernel index, so it is two axes;
    a 3x3 one overlaps the next and stays a window."""
    shapes = {"X": (1, 4, 8, 8), "W": (6, 4, kernel, kernel)}
    parser = _parse_conv(shapes, {"strides": [2, 2], "pads": [kernel // 2] * 4 if kernel % 2 else [0] * 4})
    (conv,) = parser.workload.get_computation_nodes()
    estimator = ZigZagCostEstimator(workload=parser.workload, accelerator=None, mapping=None)  # type: ignore[arg-type]
    assert len(estimator.create_equation_and_dimension_relations_and_pr_sizes(conv)[1]) == relations


def test_a_depthwise_conv_reads_the_channel_its_producer_writes():
    """A group of one channel is the channel the producer wrote, one dim of the workload, so a tile of the channels
    holds the input channels it reads."""
    weights = [
        helper.make_tensor("W0", TensorProto.FLOAT, [8, 4, 3, 3], [0.0] * 8 * 4 * 9),
        helper.make_tensor("W1", TensorProto.FLOAT, [8, 1, 3, 3], [0.0] * 8 * 9),
    ]
    window = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}
    nodes = [
        helper.make_node("Conv", ["X", "W0"], ["Y"], name="Conv", **window),
        helper.make_node("Conv", ["Y", "W1"], ["Z"], name="Depthwise", group=8, **window),
    ]
    output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "g", [_vi("X", (1, 4, 8, 8))], [output], weights)
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    workload = parser.workload
    conv, depthwise = workload.get_computation_nodes()
    assert workload.get_dims(conv)[6] in workload.get_dims(depthwise)


def _grouped_conv_on_eyeriss(channels: int, multiplier: int):
    """A 3x3 conv of ``channels`` groups of one input channel and ``multiplier`` outputs each, on an Eyeriss core."""
    shapes = {"X": (1, channels, 12, 12), "W": (channels * multiplier, 1, 3, 3)}
    parser = _parse_conv(shapes, {"group": channels, "pads": [1, 1, 1, 1]})
    (conv,) = parser.workload.get_computation_nodes()
    accelerator = parse_accelerator("stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml")
    return ZigZagCostEstimator(parser.workload, accelerator, Mapping()), conv, accelerator.get_core(0)


def test_zigzag_costs_a_depthwise_conv():
    estimator, conv, core = _grouped_conv_on_eyeriss(16, 2)
    entry = estimator.estimate(conv, core)
    assert entry.metadata["backend"] == "zigzag"
    assert entry.energy_total > 0


def test_a_conv_zigzag_cannot_cost_is_charged_its_operations(monkeypatch):
    """The ideal-cycle estimate charges the array's energy per operation, a floor, not nothing."""
    estimator, conv, core = _grouped_conv_on_eyeriss(16, 1)

    def unmapped(*_):
        raise RuntimeError("no valid loop ordering")

    monkeypatch.setattr(estimator, "run_zigzag", unmapped)
    entry = estimator.estimate(conv, core)
    assert entry.metadata["backend"] == IDEAL_CYCLE_BACKEND
    assert entry.energy_total == 16 * 12 * 12 * 3 * 3 * core.operational_array.unit.energy_cost


POOLS = {
    "strides_per_axis": ((1, 2, 9, 8), {"kernel_shape": [3, 2], "strides": [2, 1], "pads": [1, 0, 1, 0]}, (1, 0)),
    "dilations_per_axis": (
        (1, 2, 9, 9),
        {"kernel_shape": [2, 3], "strides": [1, 2], "dilations": [2, 1], "pads": [1, 1, 1, 1]},
        (1, 1),
    ),
    "window_past_the_end": ((1, 2, 8, 8), {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, (1, 1)),
}


@pytest.mark.parametrize(("x_shape", "attrs", "padding"), POOLS.values(), ids=POOLS)
def test_maxpool_access_maps_compute_a_direct_max_pool2d(x_shape, attrs, padding):
    """Every output is the largest input its iteration points read, through the same per-axis window as a Conv's:
    that is a direct max_pool2d."""
    parser = _parse("MaxPool", {"X": x_shape}, attrs)
    (pool,) = parser.workload.get_computation_nodes()
    x = np.random.default_rng(0).standard_normal(x_shape)
    sizes = [parser.workload.get_dimension_size(d) for d in parser.workload.get_dims(pool)]
    points = np.indices(sizes).reshape(len(sizes), -1).T
    index = [points @ t.A.T + t.b for t in map(AffineTransform.from_affine_map, pool.operand_mapping)]
    inside = ((index[0] >= 0) & (index[0] < x.shape)).all(1)
    found = np.full(pool.outputs[0].shape, -np.inf)
    np.maximum.at(found, tuple(index[1][inside].T), x[tuple(index[0][inside].T)])
    pads = [(p, p) for p in padding]
    windows = _windows(x, attrs["kernel_shape"], attrs.get("strides", 1), attrs.get("dilations", 1), pads, -np.inf)
    np.testing.assert_allclose(found, windows.max(axis=(-2, -1)))
