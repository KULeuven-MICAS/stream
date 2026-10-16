"""Element types: a quantized (QDQ) model runs its matmuls on the quantized tensors, a conversion the graph cannot fold
away is costed like any elementwise op, and a matrix unit keeps its partial sums at the precision it accumulates in."""

import copy
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
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
