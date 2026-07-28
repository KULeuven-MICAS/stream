"""A constant input is staged in a memory tile only when it is read more than once.

Staging costs a second transfer of the same data, so it pays for itself only when a
destination reads the tensor repeatedly. A convolution or GEMM does: it sweeps output
positions with the same operands. An elementwise node does not: it reads each element
once, and the memory tile would double its offchip traffic for nothing.
"""

from __future__ import annotations

import tempfile

import onnx
import pytest
from onnx import TensorProto, helper

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.node import ComputationNode, InEdge
from stream.workload.utils import is_reused_on_chip

_CONV = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"


def _value(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _parse(nodes, inputs, outputs):
    graph = helper.make_graph(nodes, "g", inputs, outputs, [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser.workload


def _offchip_inputs(workload):
    """Each tensor arriving from offchip, with the computation nodes that read it."""
    for node in workload.dataflow_sort():
        if not isinstance(node, InEdge):
            continue
        for tensor in node.outputs:
            consumers = [c for c in workload.successors(node) if isinstance(c, ComputationNode)]
            if consumers:
                yield tensor, consumers


@pytest.fixture(scope="module")
def convolution():
    parser = ONNXModelParser(_CONV)
    parser.run()
    return parser.workload


@pytest.fixture(scope="module")
def elementwise():
    return _parse(
        [helper.make_node("Mul", ["A", "B"], ["C"], name="mul")],
        [_value("A", (256, 2048)), _value("B", (256, 2048))],
        [_value("C", (256, 2048))],
    )


def test_convolution_operands_are_reused(convolution):
    """Every operand is swept by an iteration dimension it does not span."""
    operands = list(_offchip_inputs(convolution))
    assert operands
    for tensor, consumers in operands:
        assert is_reused_on_chip(convolution, tensor, consumers), tensor.name


def test_elementwise_operands_are_not_reused(elementwise):
    """Both operands span the whole iteration space, so each element is read once."""
    operands = list(_offchip_inputs(elementwise))
    assert len(operands) == 2
    for tensor, consumers in operands:
        assert not is_reused_on_chip(elementwise, tensor, consumers), tensor.name


def test_a_tensor_with_no_destination_is_not_reused(convolution):
    tensor, _ = next(iter(_offchip_inputs(convolution)))
    assert not is_reused_on_chip(convolution, tensor, [])
