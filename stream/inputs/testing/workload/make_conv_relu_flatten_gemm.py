import os
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


@dataclass
class ConvReluFlattenGemmConfig:
    """Configuration for a Conv -> Relu -> Flatten -> Gemm workload.

    The Flatten node triggers FusionEdge parsing, splitting the workload into
    2 fusion groups: (Conv, Relu) and (Gemm). This provides a fast multi-group
    test without the overhead of a full ResNet18 pipeline.
    """

    batch_size: int = 1
    in_channels: int = 8
    height: int = 16
    width: int = 16
    out_channels: int = 16
    kernel_size: int = 3
    gemm_out_features: int = 10
    in_dtype: str = "bf16"
    weight_dtype: str = "bf16"


def make_conv_relu_flatten_gemm_workload(config: ConvReluFlattenGemmConfig | None = None) -> str:
    """Build an ONNX model: Conv -> Relu -> Flatten -> Gemm.

    The Flatten node is parsed as a FusionEdge by the ONNX parser, which triggers
    split_fusion_groups() to produce 2 sub-workloads:
      - Group 0: Conv + Relu
      - Group 1: Gemm

    Args:
        config: Workload dimensions. Uses defaults if None.

    Returns:
        Path to the saved ONNX model.
    """
    if config is None:
        config = ConvReluFlattenGemmConfig()

    batch_size = int(config.batch_size)
    in_channels = int(config.in_channels)
    height = int(config.height)
    width = int(config.width)
    out_channels = int(config.out_channels)
    kernel_size = int(config.kernel_size)
    gemm_out_features = int(config.gemm_out_features)

    # Data type mapping
    dtype_map = {
        "f32": TensorProto.FLOAT,
        "f16": TensorProto.FLOAT16,
        "bf16": TensorProto.BFLOAT16,
        "i8": TensorProto.INT8,
    }
    if config.in_dtype not in dtype_map or config.weight_dtype not in dtype_map:
        raise ValueError("Unsupported data type. Supported types: f32, f16, bf16, i8.")

    in_tensor_type = dtype_map[config.in_dtype]
    weight_tensor_type = dtype_map[config.weight_dtype]

    # Compute intermediate shapes
    pad = kernel_size // 2
    # Conv output: same spatial dims with padding
    conv_out_h = height
    conv_out_w = width
    flatten_size = out_channels * conv_out_h * conv_out_w

    # Initializers (weights)
    conv_weights = helper.make_tensor(
        "conv_weights",
        weight_tensor_type,
        [out_channels, in_channels, kernel_size, kernel_size],
        np.zeros((out_channels, in_channels, kernel_size, kernel_size)),
    )
    conv_weights.ClearField("float_data")

    gemm_weights = helper.make_tensor(
        "gemm_weights",
        weight_tensor_type,
        [flatten_size, gemm_out_features],
        np.zeros((flatten_size, gemm_out_features)),
    )
    gemm_weights.ClearField("float_data")

    # Graph I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, in_channels, height, width])
    out = helper.make_tensor_value_info("output", in_tensor_type, [batch_size, gemm_out_features])

    # Nodes
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weights"],
        outputs=["conv_out"],
        name="Conv1",
        kernel_shape=[kernel_size, kernel_size],
        pads=[pad, pad, pad, pad],
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_out"],
        outputs=["relu_out"],
        name="Relu1",
    )

    flatten_node = helper.make_node(
        "Flatten",
        inputs=["relu_out"],
        outputs=["flatten_out"],
        name="Flatten1",
        axis=1,
    )

    gemm_node = helper.make_node(
        "Gemm",
        inputs=["flatten_out", "gemm_weights"],
        outputs=["output"],
        name="Gemm1",
    )

    graph = helper.make_graph(
        nodes=[conv_node, relu_node, flatten_node, gemm_node],
        name="ConvReluFlattenGemm",
        inputs=[inp],
        outputs=[out],
        initializer=[conv_weights, gemm_weights],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Conv -> Relu -> Flatten -> Gemm. Flatten triggers FusionEdge split into 2 groups.",
    )

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    onnx_path = os.path.join(
        os.path.dirname(__file__),
        f"conv_relu_flatten_gemm_{batch_size}_{in_channels}_{height}_{width}_{out_channels}_{gemm_out_features}.onnx",
    )
    onnx.save(inferred, onnx_path)
    print(f"Conv-Relu-Flatten-Gemm ONNX model created: {onnx_path}")

    return onnx_path
