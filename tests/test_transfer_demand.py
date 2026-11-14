"""What the steady-state IR says a transfer sends per source, target and kind, and the bits each link carries."""

import tempfile
from pathlib import Path
from typing import Any

from conv_windows import HARDWARE, conv, onnx_graph
from onnx import helper

from stream.api import SolveOptions, evaluate_mapping
from stream.ir import AllocationIR

TPU = HARDWARE.format("tpu_like_quad_core")


def _transfer(workload: str, mapping: str, name: str) -> dict[str, Any]:
    """The steady-state detail of the transfer ``name`` under ``mapping`` on the quad-core TPU."""
    with tempfile.TemporaryDirectory() as out:
        path = Path(out) / "mapping.yaml"
        path.write_text(mapping)
        allocation = evaluate_mapping(TPU, workload, out, str(path), SolveOptions(artifacts=False)).context.get(
            "allocation"
        )
    nodes = AllocationIR.from_internal(allocation).model_dump()["steady_state"]["tiled_graph"]["nodes"]
    return next(node for node in nodes if node["name"] == name)


def _demand_by_kind(transfer: dict[str, Any]) -> dict[tuple[str, str, str], int | float]:
    return {(h["source"], h["target"], h["kind"]): h["elements"] for h in transfer["demand"]}


def _core(i: int) -> str:
    return f"tpu_like_{i}"


SPLIT_COLUMNS = """layers:
- {name: Conv1, core_allocation: [[0, 1, 2, 3]], inter_core_tiling: [[{dim: D1, split: 4}]]}
- {name: Conv2, core_allocation: [[0, 1, 2, 3]], inter_core_tiling: [[{dim: D1, split: 4}]]}
fused_groups:
- {name: G, layers: [Conv1, Conv2]}
"""


def test_each_link_reports_the_bits_its_route_cycles_price():
    """Two 3x3 convs of 8 channels split by columns over four cores: each core sends each neighbour one column of
    conv1_out, 8*8 = 64 elements, 1024 bits over a 32-bit link in 32 cycles, and keeps its own 4 columns, 256."""
    nodes = [conv("Conv1", "input", "w1", "conv1_out"), conv("Conv2", "conv1_out", "w2", "out")]
    workload = onnx_graph(nodes, {"input": [1, 4, 8, 16], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])
    transfer = _transfer(workload, SPLIT_COLUMNS, "Transfer(conv1_out)")
    neighbours = {(_core(i), _core(j), "moved"): 64 for i in range(4) for j in (i - 1, i + 1) if 0 <= j < 4}
    assert _demand_by_kind(transfer) == neighbours | {(_core(i), _core(i), "in_place"): 256 for i in range(4)}
    bits = dict(zip(transfer["links"], transfer["link_bits"], strict=True))
    assert bits["tpu_like_0->tpu_like_1"] == 1024
    assert transfer["route_cycles"] == 32


CONTRACTION_SPLIT = """layers:
- {name: Gemm, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D2, split: 2}]]}
fused_groups:
- {name: G, layers: [Gemm]}
"""


def test_a_reduction_over_cores_reports_each_cores_partial_sum():
    """Y = X W with the 64-long contraction split over cores 0 and 1: each sends its partial sum of Y, 16*32 = 512
    elements, and Y is delivered once, 8192 bits. Each core sends the other the half it completes at the 32-bit
    accumulator, 256*32 = 8192 bits, then its finished half off chip at bf16, 4096 bits: 24576 bits on the 128-bit
    bus, 192 cycles."""
    gemm = helper.make_node("MatMul", ["X", "W"], ["Y"], name="Gemm")
    workload = onnx_graph([gemm], {"X": [16, 64], "W": [64, 32]}, ["Y"])
    transfer = _transfer(workload, CONTRACTION_SPLIT, "Transfer(Y_1)")
    assert _demand_by_kind(transfer) == {
        (_core(0), "offchip", "partial_sum"): 512,
        (_core(1), "offchip", "partial_sum"): 512,
    }
    assert transfer["moved_bits"] == 8192
    assert transfer["link_bits"] == [24576] and transfer["route_cycles"] == 192


HEAD_MAPPING = """layers:
- {name: ProjQ, core_allocation: [[0]]}
- {name: ProjK, core_allocation: [[0]]}
- {name: ProjV, core_allocation: [[0]]}
- {name: Transpose, core_allocation: [[0]]}
- {name: QK, core_allocation: [[0]]}
- {name: Softmax, core_allocation: [[0]]}
- {name: PV, core_allocation: [[0]]}
fused_groups:
- {name: G, layers: [ProjQ, ProjK, ProjV, QK, Softmax, PV], intra_core_tiling: [{dim: QK.D0, tile: 16}]}
"""


def test_a_local_copy_is_reported_as_local_copy():
    """An attention head on core 0 in query tiles of 16 rows: ProjK and ProjV read all of I (64,64), 4096 elements
    from off chip; the 16*64 = 1024 elements of the query tile ProjQ reads are copied from it in core 0's memory."""
    nodes = [
        helper.make_node("MatMul", ["I", w], [y], name=name)
        for name, w, y in (("ProjQ", "wQ", "Q"), ("ProjK", "wK", "K"), ("ProjV", "wV", "V"))
    ] + [
        helper.make_node("Transpose", ["K"], ["KT"], name="Transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["Q", "KT"], ["scores"], name="QK"),
        helper.make_node("Softmax", ["scores"], ["attn"], name="Softmax", axis=-1),
        helper.make_node("MatMul", ["attn", "V"], ["Y"], name="PV"),
    ]
    workload = onnx_graph(nodes, {"I": [64, 64], "wQ": [64, 32], "wK": [64, 32], "wV": [64, 32]}, ["Y"])
    transfer = _transfer(workload, HEAD_MAPPING, "Transfer(I)")
    assert _demand_by_kind(transfer) == {("offchip", _core(0), "moved"): 4096, (_core(0), _core(0), "local_copy"): 1024}
