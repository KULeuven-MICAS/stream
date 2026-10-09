"""The cycles a group reports are the latency its solve models: the latency objective level, in which every iteration
is held to its busiest core, link and memory port."""

import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def _matmul(path: Path) -> str:
    """A 256 x 2048 x 2048 bf16 matmul, whose 8 MB of weights stream from off-chip memory."""
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "w"], ["c"], name="MatMul")],
        "matmul",
        [
            helper.make_tensor_value_info("a", TensorProto.BFLOAT16, [256, 2048]),
            helper.make_tensor_value_info("w", TensorProto.BFLOAT16, [2048, 2048]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.BFLOAT16, [256, 2048])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return str(path)


def test_a_streaming_node_reports_the_latency_its_solve_models_held_to_its_busiest_port():
    with tempfile.TemporaryDirectory() as tmp:
        estimate = evaluate_mapping(ACCELERATOR, _matmul(Path(tmp) / "matmul.onnx"), tmp, options=SolveOptions())
    allocation = estimate.context.get("allocation")
    (cycles,) = estimate.group_cycles
    assert cycles == allocation.solution.latency.total == pytest.approx(allocation.solution.primary_cost)
    rows = allocation.get_ir()["performance"]["memory_ports"]
    assert cycles >= max(row["real_cycle"] for row in rows) * allocation.problem.iterations
    assert max(row["utilization"] for row in rows) <= 1 + 1e-6
