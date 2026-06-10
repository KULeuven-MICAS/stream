import os
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def _clear_tensor_data(tensor: TensorProto) -> None:
    """Remove raw weight data from a tensor, keeping only shape and type metadata.

    Clears every possible data field -- bf16 weights are packed into int32_data, so clearing
    only float_data would leave the values in place and bloat the saved ONNX.
    """
    for field in ("float_data", "double_data", "int32_data", "int64_data", "uint64_data", "raw_data"):
        tensor.ClearField(field)


@dataclass
class TwoConvWorkloadConfig:
    batch_size: int
    height: int
    width: int
    in_channels: int
    out_channels_1: int
    out_channels_2: int
    kernel_size: int
    in_dtype: str
    weight_dtype: str


def make_2_conv_workload(config: TwoConvWorkloadConfig):
    """
    Build an ONNX model with two Conv operators.

    Conv1: [batch_size, in_channels, height, width] -> [batch_size, out_channels_1, height, width]
    Conv2: [batch_size, out_channels_1, height, width] -> [batch_size, out_channels_2, height, width]

    Args:
        config (TwoConvWorkloadConfig): Configuration for the two-conv workload.

    Returns:
        str: Path to the saved ONNX model.
    """
    # Extract values from config
    batch_size = int(config.batch_size)
    height = int(config.height)
    width = int(config.width)
    in_channels = int(config.in_channels)
    out_channels_1 = int(config.out_channels_1)
    out_channels_2 = int(config.out_channels_2)
    kernel_size = int(config.kernel_size)
    in_dtype = config.in_dtype
    weight_dtype = config.weight_dtype

    # Data type mapping
    dtype_map = {
        "f32": TensorProto.FLOAT,
        "f16": TensorProto.FLOAT16,
        "bf16": TensorProto.BFLOAT16,
        "i8": TensorProto.INT8,
    }
    if in_dtype not in dtype_map or weight_dtype not in dtype_map:
        raise ValueError("Unsupported data type. Supported types: f32, f16, bf16, i8.")

    in_tensor_type = dtype_map[in_dtype]
    weight_tensor_type = dtype_map[weight_dtype]

    # Initializers (weights)
    w1 = helper.make_tensor(
        "weights_1",
        weight_tensor_type,
        [out_channels_1, in_channels, kernel_size, kernel_size],
        np.zeros((out_channels_1, in_channels, kernel_size, kernel_size)),
    )
    _clear_tensor_data(w1)

    w2 = helper.make_tensor(
        "weights_2",
        weight_tensor_type,
        [out_channels_2, out_channels_1, kernel_size, kernel_size],
        np.zeros((out_channels_2, out_channels_1, kernel_size, kernel_size)),
    )
    _clear_tensor_data(w2)

    # I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, in_channels, height, width])
    out = helper.make_tensor_value_info("output", in_tensor_type, [batch_size, out_channels_2, height, width])

    # Conv nodes
    pad = kernel_size // 2  # To maintain spatial dimensions
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "weights_1"],
        outputs=["conv1_out"],
        name="Conv1",
        kernel_shape=[kernel_size, kernel_size],
        pads=[pad, pad, pad, pad],
    )

    conv2 = helper.make_node(
        "Conv",
        inputs=["conv1_out", "weights_2"],
        outputs=["output"],
        name="Conv2",
        kernel_shape=[kernel_size, kernel_size],
        pads=[pad, pad, pad, pad],
    )

    graph = helper.make_graph(
        nodes=[conv1, conv2],
        name="TwoConv",
        inputs=[inp],
        outputs=[out],
        initializer=[w1, w2],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Two Conv operators in sequence.",
    )

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    onnx_path = os.path.join(
        os.path.dirname(__file__),
        f"2conv_{batch_size}_{in_channels}_{height}_{width}_{out_channels_1}_{out_channels_2}_{kernel_size}.onnx",
    )
    onnx.save(inferred, onnx_path)
    print(f"2-Conv ONNX model created: {onnx_path}")

    return onnx_path
