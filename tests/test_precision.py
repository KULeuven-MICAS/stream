"""Element types: a quantized (QDQ) model runs its matmuls on the quantized tensors, a conversion the graph cannot fold
away is costed like any elementwise op, and a matrix unit keeps its partial sums at the precision it accumulates in."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from conv_windows import onnx_graph
from onnx import TensorProto, helper, numpy_helper
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.ports import OUTPUT, READ, READ_BY_DATAPATH, WRITE_BY_DATAPATH
from stream.ir import AllocationIR
from stream.opt.allocation.constraint_optimization.families.traffic import _running_sum_traffic
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_backends import entry_backend, port_traffic
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.workload import ComputationNode, InEdge

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
MXU = "stream/inputs/examples/hardware/cores/tpu_v7_mxu.yaml"


def _save(nodes, inputs, output, initializers, path):
    graph = helper.make_graph(nodes, "graph", inputs, [output], initializers)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), path)
    return str(path)


def _qdq_conv(path):
    """A convolution as a quantizer exports it: int8 activations and weights and an int32 bias, each dequantized
    before it is read, and the result quantized to int8, in a model that takes and returns fp32."""
    initializers = [
        numpy_helper.from_array(np.array(0.1, np.float32), "scale"),
        numpy_helper.from_array(np.array(0, np.int8), "zero"),
        numpy_helper.from_array(np.array(0, np.int32), "zero_32"),
        numpy_helper.from_array(np.ones((16, 8, 3, 3), np.int8), "w_q"),
        numpy_helper.from_array(np.ones(16, np.int32), "b_q"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"], name="Quantize_x"),
        helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["x_dq"], name="Dequantize_x"),
        helper.make_node("DequantizeLinear", ["w_q", "scale", "zero"], ["w_dq"], name="Dequantize_w"),
        helper.make_node("DequantizeLinear", ["b_q", "scale", "zero_32"], ["b_dq"], name="Dequantize_b"),
        helper.make_node("Conv", ["x_dq", "w_dq", "b_dq"], ["y"], name="Conv", pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["y", "scale", "zero"], ["y_q"], name="Quantize_y"),
        helper.make_node("DequantizeLinear", ["y_q", "scale", "zero"], ["out"], name="Dequantize_y"),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 16, 16, 16])
    return _save(nodes, [x], out, initializers, path)


def _workload(path):
    ctx = StageContext.from_kwargs(accelerator=TPU_V7, workload_path=path, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0].get("workload")


def _types(tensors):
    return [str(t.operand_type) for t in tensors]


def test_a_quantized_model_runs_its_convolution_on_the_quantized_tensors(tmp_path):
    """The dequantizations fold into the convolution, which reads int8 and the int32 bias, and the quantization of its
    result folds into it too, so it writes int8. Quantizing the fp32 input and dequantizing the output have nothing to
    fold into and stay conversions, which read neither the scale nor the zero point."""
    workload = _workload(_qdq_conv(tmp_path / "qdq.onnx"))
    nodes = {n.name: n for n in workload.nodes if isinstance(n, ComputationNode)}
    assert {name: n.type for name, n in nodes.items()} == {
        "Quantize_x": "Cast",
        "Conv": "Conv",
        "Dequantize_y": "Cast",
    }
    assert _types(nodes["Conv"].inputs) == ["i8", "i8", "i32"]
    assert _types(nodes["Conv"].outputs) == ["i8"]
    assert _types(nodes["Quantize_x"].outputs) == ["i8"]
    assert _types(nodes["Dequantize_y"].outputs) == ["f32"]
    assert {n.name for n in workload.nodes if isinstance(n, InEdge)} == {"x", "w_q", "b_q"}


def test_a_cast_converts_a_tensor_to_another_element_type(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 64])
    nodes = [
        helper.make_node("Cast", ["x"], ["x_bf16"], name="Cast_bf16", to=TensorProto.BFLOAT16),
        helper.make_node("Relu", ["x_bf16"], ["y"], name="Relu"),
        helper.make_node("Cast", ["y"], ["out"], name="Cast_fp16", to=TensorProto.FLOAT16),
    ]
    workload = _workload(_save(nodes, [x], out, [], tmp_path / "cast.onnx"))
    nodes = {n.name: n for n in workload.nodes if isinstance(n, ComputationNode)}
    assert [nodes[name].type for name in ("Cast_bf16", "Relu", "Cast_fp16")] == ["Cast", "Relu", "Cast"]
    assert _types(nodes["Relu"].inputs) == ["bf16"]
    assert _types(nodes["Cast_fp16"].outputs) == ["f16"]


def _matmul(element_type, path):
    a = helper.make_tensor_value_info("a", element_type, [64, 128])
    b = helper.make_tensor_value_info("b", element_type, [128, 32])
    out = helper.make_tensor_value_info("out", element_type, [64, 32])
    return _save([helper.make_node("MatMul", ["a", "b"], ["out"], name="MatMul")], [a, b], out, [], path)


@pytest.mark.parametrize(
    ("element_type", "bits"), [("INT4", 4), ("UINT4", 4), ("FLOAT4E2M1", 4), ("FLOAT8E4M3FN", 8), ("FLOAT8E5M2", 8)]
)
def test_narrow_element_types_are_read_at_their_width(element_type, bits, tmp_path):
    workload = _workload(_matmul(TensorProto.DataType.Value(element_type), tmp_path / "matmul.onnx"))
    (matmul,) = [n for n in workload.nodes if isinstance(n, ComputationNode)]
    elements = [64 * 128, 128 * 32, 64 * 32]
    assert [t.size_bits() for t in (*matmul.inputs, *matmul.outputs)] == [n * bits for n in elements]


def test_the_mxu_keeps_partial_sums_at_the_precision_it_accumulates_in(tmp_path):
    """The MXU declares an fp32 accumulator: the convolution's partial sums are 32 bits wide and its result is written
    as int8. The vector unit's conversions are not accumulations, so they carry no wider partial sums."""
    estimate = evaluate_mapping(
        TPU_V7, _qdq_conv(tmp_path / "qdq.onnx"), str(tmp_path), options=SolveOptions(artifacts=False)
    )
    precisions = {
        node.type: {str(op): bits for op, bits in entry.cme.layer.operand_precision.data.items()}
        for node, costs in estimate.context.get("cost_lut").lut.items()
        for entry in costs.values()
    }
    assert precisions["Conv"]["O"] == 32
    assert precisions["Conv"]["O_final"] == 8
    assert precisions["Cast"]["O"] == precisions["Cast"]["O_final"]


def test_a_core_names_its_precision_in_known_element_types():
    validator = AcceleratorValidator(open_yaml(TPU_V7), accelerator_path=TPU_V7)
    core = open_yaml(MXU)
    core["operand_precision"] = {"input": "bf16", "accumulator": "fp33"}
    validator.validate_core_data(core, "mxu")
    assert any("fp33" in error for error in validator.errors)


@pytest.mark.parametrize("element_type", ["uint4", "fp4", "fp8", "fp8_e5m2"])
def test_a_core_may_name_a_narrow_precision(element_type):
    validator = AcceleratorValidator(open_yaml(TPU_V7), accelerator_path=TPU_V7)
    core = open_yaml(MXU)
    core["operand_precision"] = {"input": element_type, "accumulator": "fp32"}
    validator.validate_core_data(core, "mxu")
    assert not validator.errors


def test_a_quantized_linear_layer_solves_although_gemm_does_not_read_its_bias(tmp_path):
    """The Gemm reads its int8 operands; its int32 bias, which the Gemm parser leaves out, is no dangling input."""
    initializers = [
        numpy_helper.from_array(np.array(0.1, np.float32), "scale"),
        numpy_helper.from_array(np.array(0, np.int8), "zero"),
        numpy_helper.from_array(np.array(0, np.int32), "zero_32"),
        numpy_helper.from_array(np.ones((256, 256), np.int8), "w_q"),
        numpy_helper.from_array(np.ones(256, np.int32), "b_q"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["x"], name="Dequantize_x"),
        helper.make_node("DequantizeLinear", ["w_q", "scale", "zero"], ["w"], name="Dequantize_w"),
        helper.make_node("DequantizeLinear", ["b_q", "scale", "zero_32"], ["b"], name="Dequantize_b"),
        helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="Gemm"),
        helper.make_node("QuantizeLinear", ["y", "scale", "zero"], ["y_q"], name="Quantize_y"),
    ]
    x = helper.make_tensor_value_info("x_q", TensorProto.INT8, [256, 256])
    out = helper.make_tensor_value_info("y_q", TensorProto.INT8, [256, 256])
    path = _save(nodes, [x], out, initializers, tmp_path / "linear.onnx")
    workload = _workload(path)
    assert {n.name for n in workload.nodes if isinstance(n, InEdge)} == {"x_q", "w_q"}
    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    assert estimate.cycles > 0


def test_a_reshape_does_not_requantize(tmp_path):
    """A layout-only node does not compute, so the quantization of its output stays a conversion of its own."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out", TensorProto.INT8, [64])
    initializers = [
        numpy_helper.from_array(np.array(0.1, np.float32), "scale"),
        numpy_helper.from_array(np.array(0, np.int8), "zero"),
        numpy_helper.from_array(np.array([64], np.int64), "shape"),
    ]
    nodes = [
        helper.make_node("Relu", ["x"], ["y"], name="Relu"),
        helper.make_node("Reshape", ["y", "shape"], ["y_flat"], name="Reshape"),
        helper.make_node("QuantizeLinear", ["y_flat", "scale", "zero"], ["out"], name="Quantize"),
    ]
    workload = _workload(_save(nodes, [x], out, initializers, tmp_path / "reshape.onnx"))
    nodes = {n.name: n for n in workload.nodes if isinstance(n, ComputationNode)}
    assert nodes["Quantize"].type == "Cast"
    assert _types(nodes["Relu"].outputs) == ["f32"]


def test_cores_with_different_accumulators_do_not_share_costs():
    ctx = StageContext.from_kwargs(accelerator=TPU_V7, workload_path=None, output_path=tempfile.mkdtemp())
    mxu = MainStage([AcceleratorParserStage, LeafStage], ctx).run()[0].get("accelerator").get_core(0)
    narrow = copy.copy(mxu)
    narrow.operand_precision = {"input": "int8", "accumulator": "int16"}
    assert mxu.has_same_performance(copy.copy(mxu))
    assert not mxu.has_same_performance(narrow)


FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
TPU_QUAD = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
EYERISS = "stream/inputs/examples/hardware/eyeriss_like_single_core.yaml"


def _mlp(path):
    """X[64,128] @ W1[128,256] -> H1[64,256] @ W2[256,128] -> Y[64,128], all bf16."""
    initializers = [
        numpy_helper.from_array(np.zeros((128, 256), np.float32), "W1"),
        numpy_helper.from_array(np.zeros((256, 128), np.float32), "W2"),
    ]
    for weight in initializers:
        weight.data_type = TensorProto.BFLOAT16
        weight.ClearField("float_data")
        weight.ClearField("raw_data")
        weight.int32_data.extend([0] * int(np.prod(weight.dims)))
    nodes = [
        helper.make_node("Gemm", ["X", "W1"], ["H1"], name="Gemm1"),
        helper.make_node("Gemm", ["H1", "W2"], ["Y"], name="Gemm2"),
    ]
    x = helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [64, 128])
    out = helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [64, 128])
    return _save(nodes, [x], out, initializers, path)


def _estimated_mlp(hardware, mapping, tmp_path):
    (tmp_path / "mapping.yaml").write_text(mapping)
    return evaluate_mapping(
        hardware,
        _mlp(tmp_path / "mlp.onnx"),
        str(tmp_path),
        mapping=str(tmp_path / "mapping.yaml"),
        options=SolveOptions(artifacts=False),
    )


def _solved_mlp(hardware, mapping, tmp_path):
    (ir,) = _estimated_mlp(hardware, mapping, tmp_path).context.data["group_allocations"].values()
    return ir


_FUSED_HIDDEN = """
layers:
- {name: Gemm1, core_allocation: [[0]]}
- {name: Gemm2, core_allocation: [[0]]}
fused_groups:
- {name: G, layers: [Gemm1, Gemm2], intra_core_tiling: [{dim: Gemm1.D2, tile: 64}]}
"""


def test_a_running_sum_across_a_fused_loop_is_held_at_the_accumulator_precision(tmp_path):
    """Fused on the hidden dim (tile 64, 4 iterations), Gemm2's output Y_1[64,128] is a running sum between
    iterations, held at fusemax's fp32 accumulator, 64 x 128 x 32 bits; the finished Y off chip keeps bf16."""
    ir = _solved_mlp(FUSEMAX, _FUSED_HIDDEN, tmp_path)
    tensors = {t["name"]: t["bits"] for t in ir["steady_state"]["tensors"]}
    occupancy = {
        (m["memory"], t["tensor"]): t["bits"] for m in ir["performance"]["memory_occupancy"] for t in m["tensors"]
    }
    assert tensors["Y_1"] == occupancy[("sram", "Y_1")] == 64 * 128 * 32
    assert tensors["Y"] == occupancy[("dram", "Y")] == 64 * 128 * 16


_SPLIT_REDUCTION = """
layers:
- {name: Gemm1, core_allocation: [[0, 1, 2, 3]], inter_core_tiling: [[{dim: D2, split: 4}]]}
- {name: Gemm2, core_allocation: [[0, 1, 2, 3]], inter_core_tiling: [[{dim: D1, split: 4}]]}
fused_groups:
- {name: G, layers: [Gemm1, Gemm2], intra_core_tiling: [{dim: Gemm1.D0, tile: 16}]}
"""


def test_a_core_holds_its_share_of_a_split_reduction_at_the_accumulator_precision(tmp_path):
    """Gemm2 split over 4 cores on its contraction, fused on tokens (tile 16): each core holds its partial
    Y_1[16,128] at the int32 accumulator, double-buffered: 2 x 16 x 128 x 32 bits."""
    ir = _solved_mlp(TPU_QUAD, _SPLIT_REDUCTION, tmp_path)
    resident = {
        m["memory"]: t["bits"]
        for m in ir["performance"]["memory_occupancy"]
        for t in m["tensors"]
        if t["tensor"] == "Y_1"
    }
    assert resident == {f"tpu_like_{core}": 2 * 16 * 128 * 32 for core in range(4)}


def test_partial_sums_cross_cores_at_the_accumulator_precision(tmp_path):
    """Summing the 4 cores' partial Y_1[16,128] of the split above moves 3 x 16 x 128 = 6144 partial sums between
    cores per iteration at the int32 accumulator, beyond the finished sum the off-chip port takes."""
    ir = _solved_mlp(TPU_QUAD, _SPLIT_REDUCTION, tmp_path)
    ports = {a["resource"]: a["bits_per_iteration"] for a in ir["performance"]["memory_ports"]}
    links = sum(bits for resource, bits in ports.items() if "->" in resource or resource.startswith("bus "))
    assert links - ports["offchip.rw_port_1"] == 3 * 16 * 128 * 32


def test_the_cores_read_the_partial_sums_they_send_at_the_accumulator_precision(tmp_path):
    """Of the split above, the four cores send 3 x 16 x 128 = 6144 partial sums to be added up at the int32
    accumulator and write the 16 x 128 = 2048 finished sums off chip at bf16: their read ports pass
    (6144 x 32 + 2048 x 16) / (2048 x 16) = 7 times the bits the reduction delivers."""
    allocation = _estimated_mlp(TPU_QUAD, _SPLIT_REDUCTION, tmp_path).context.get("allocation")
    (tr, route) = next((tr, r) for tr, r in allocation.solution.transfer_routes.items() if tr.name == "Transfer(Y_1)")
    cores = [memory for memory in route.sources if not memory.offchip]
    sent = sum(sum(allocation.space.side_ports(tr, route, memory, READ).values()) for memory in cores)
    assert sent == pytest.approx(7)


def _output_traffic(hardware, tmp_path):
    """Gemm2's bits through its output's ports per firing, fused on the hidden dim, by direction."""
    space = _estimated_mlp(hardware, _FUSED_HIDDEN, tmp_path).context.get("allocation").space
    (gemm2,) = (n for n in space.ssc_nodes if n.name == "Gemm2")
    (core,) = space.cost_lut.get_cores(gemm2)
    entry = space.cost_lut.get_cost(gemm2, core)
    charged = [
        *port_traffic(entry_backend(entry, core), entry),
        *_running_sum_traffic(SimpleNamespace(space=space), gemm2),
    ]
    moved = {direction: 0.0 for direction in (READ_BY_DATAPATH, WRITE_BY_DATAPATH)}
    for role, direction, bits in charged:
        moved[direction] += bits if role == OUTPUT else 0
    return moved


def test_a_running_sum_is_written_and_read_back_at_the_accumulator_precision(tmp_path):
    """Fused on the hidden dim (4 firings), Gemm2 writes its 64 x 128 running sum at fp32 on 3 firings and the
    finished sum at bf16 on the last, (3 x 32 + 16) / 4 = 28 bits an element a firing, and reads the partial sum back
    at fp32 on 3 of the 4."""
    moved = _output_traffic(FUSEMAX, tmp_path)
    assert moved == {WRITE_BY_DATAPATH: 64 * 128 * 28, READ_BY_DATAPATH: 64 * 128 * 32 * 3 / 4}


def test_a_running_sum_no_wider_than_its_type_is_read_back_at_its_type(tmp_path):
    """On eyeriss_like_single_core, whose int16 accumulator is no wider than bf16, Gemm2 reads its 64 x 128 sum back
    at bf16 on 3 of its 4 firings."""
    moved = _output_traffic(EYERISS, tmp_path)
    assert moved[READ_BY_DATAPATH] == 64 * 128 * 16 * 3 / 4


def test_a_running_sum_s_extra_writes_and_read_back_cost_their_port_energy(tmp_path):
    """On fusemax, Gemm2's running sum adds 16 more bits an element written and the fp32 read-back on 3 of its 4
    firings: 64 x 128 x (16 + 32) x 3 / 4 bits a firing through sram's ports at 1 pJ a bit."""
    estimate = _estimated_mlp(FUSEMAX, _FUSED_HIDDEN, tmp_path)
    space = estimate.context.get("allocation").space
    (ir,) = estimate.context.data["group_allocations"].values()
    own = sum(
        space.cost_lut.get_cost(node, core).energy_total
        for node in space.ssc_nodes
        for (core,) in [space.cost_lut.get_cores(node)]
    )
    assert ir["performance"]["energy"]["node_pj_per_iteration"] == pytest.approx(own + 64 * 128 * 48 * 3 / 4)


def test_partial_sums_added_up_on_chip_are_laid_out_at_the_accumulator_precision(tmp_path):
    """Of the split reduction over four cores, the transfer adding up the partial Y_1 between cores lays out the int32
    accumulator's 32-bit elements."""
    ir = _solved_mlp(TPU_QUAD, _SPLIT_REDUCTION, tmp_path)
    (bits,) = (row["element_bits"] for row in ir["performance"]["layouts"] if row["transfer"] == "Transfer(Y_1)")
    assert bits == 32


def test_a_memory_of_measured_bandwidth_passes_the_partial_sums_it_sends_at_the_accumulator_precision(tmp_path):
    """Y = X W with the contraction split over cores 0 and 1, core 1's memory given a measured bandwidth: it receives
    its halves of X and W, 8192 and 16384 bits, and sends core 0 its partial Y, 16 x 32 at the int32 accumulator."""
    hardware = Path(TPU_QUAD).read_text() + "\nbandwidth:\n  tpu_like_1: {ceiling: 64, contiguous: 64}\n"
    (tmp_path / "chip.yaml").write_text(hardware)
    (tmp_path / "cores").symlink_to(Path(TPU_QUAD).parent.resolve() / "cores")
    (tmp_path / "mapping.yaml").write_text(
        """layers:
- {name: Gemm, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D2, split: 2}]]}
fused_groups:
- {name: G, layers: [Gemm]}
"""
    )
    workload = onnx_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="Gemm")], {"X": [16, 64], "W": [64, 32]}, ["Y"]
    )
    options = SolveOptions(artifacts=False)
    estimate = evaluate_mapping(
        str(tmp_path / "chip.yaml"), workload, str(tmp_path), str(tmp_path / "mapping.yaml"), options
    )
    rows = AllocationIR.from_internal(estimate.context.get("allocation")).performance.memory_ports
    (row,) = (row for row in rows if row.kind == "shared_bandwidth")
    assert (row.resource, row.bits_per_iteration) == ("tpu_like_1", 8192 + 16384 + 16 * 32 * 32)
