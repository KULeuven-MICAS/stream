"""Tests for the Gemm parser: its affine maps honour transA and transB."""

import onnx
import pytest
from onnx import TensorProto, helper

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.affine_access import footprint

M, K, N = 2, 6, 10


@pytest.mark.parametrize("trans_a", [0, 1])
@pytest.mark.parametrize("trans_b", [0, 1])
def test_gemm_reads_every_operand_whole_in_its_stored_layout(tmp_path, trans_a: int, trans_b: int):
    """Over the full (m, k, n) space every operand's footprint is its stored shape, A as [K, M] under transA and B
    as [N, K] under transB, as torchvision's classifier exports its weight."""
    a_shape, b_shape = (K, M) if trans_a else (M, K), (N, K) if trans_b else (K, N)
    node = helper.make_node("Gemm", ["A", "B"], ["C"], name="gemm", transA=trans_a, transB=trans_b)
    graph = helper.make_graph(
        [node],
        "g",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, (M, N))],
    )
    path = str(tmp_path / "gemm.onnx")
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    parser = ONNXModelParser(path)
    parser.run()
    workload = parser.workload
    (gemm,) = workload.get_computation_nodes()

    assert workload.get_dimension_sizes() == (M, K, N)
    space = {d: range(size) for d, size in enumerate(workload.get_dimension_sizes())}
    for tensor, mapping in zip(gemm.tensors, gemm.operand_mapping, strict=True):
        assert footprint(mapping, space) == tuple(map(range, tensor.shape)), tensor.name
