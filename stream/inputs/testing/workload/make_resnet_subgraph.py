import os
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


class ResNetPattern(Enum):
    """Enumeration of supported ResNet18 sub-graph patterns."""

    BASIC_RESIDUAL = auto()  # Conv->Relu->Conv->Add with skip
    STRIDE2_DOWNSAMPLE = auto()  # Conv(s=2)->Relu->Conv + Conv(1x1,s=2)->Add
    FRONTEND = auto()  # Conv(7x7,s=2)->Relu->MaxPool(s=2)
    DUAL_RESIDUAL = auto()  # Two residual blocks + Reshape FusionEdge between


@dataclass
class ResNetSubgraphConfig:
    """Configuration for a ResNet18 sub-graph pattern.

    Default dimensions are small for fast testing. Some patterns override
    defaults internally to ensure valid spatial arithmetic (e.g., STRIDE2_DOWNSAMPLE
    uses height=32, width=32 for stride-2 Conv).
    """

    pattern: ResNetPattern
    batch_size: int = 1
    in_channels: int = 8
    height: int = 16
    width: int = 16
    out_channels: int = 8
    in_dtype: str = "bf16"
    weight_dtype: str = "bf16"


# Data type mapping (shared across all patterns)
_DTYPE_MAP: dict[str, int] = {
    "f32": TensorProto.FLOAT,
    "f16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
    "i8": TensorProto.INT8,
}


def _make_weight(name: str, tensor_type: int, shape: list[int]) -> onnx.TensorProto:
    """Create a zero-filled weight initializer with cleared float_data."""
    w = helper.make_tensor(name, tensor_type, shape, np.zeros(shape))
    w.ClearField("float_data")
    return w


def _build_basic_residual(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    in_tensor_type: int,
    weight_tensor_type: int,
) -> onnx.ModelProto:
    """Build BASIC_RESIDUAL: Conv->Relu->Conv->Add with skip connection.

    The input tensor fans out to both Conv1 (main path) and Add1 (skip connection).
    out_channels == in_channels so skip dimensions match.
    """
    c = in_channels

    # Weights
    w1 = _make_weight("w1", weight_tensor_type, [c, c, 3, 3])
    w2 = _make_weight("w2", weight_tensor_type, [c, c, 3, 3])

    # I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, c, height, width])
    out = helper.make_tensor_value_info("add1_out", in_tensor_type, [batch_size, c, height, width])

    # Nodes
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        name="Conv1",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"], name="Relu1")
    conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "w2"],
        outputs=["conv2_out"],
        name="Conv2",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    add1 = helper.make_node(
        "Add",
        inputs=["conv2_out", "input"],
        outputs=["add1_out"],
        name="Add1",
    )

    graph = helper.make_graph(
        nodes=[conv1, relu1, conv2, add1],
        name="BasicResidual",
        inputs=[inp],
        outputs=[out],
        initializer=[w1, w2],
    )

    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Basic residual block: Conv->Relu->Conv->Add with skip connection.",
    )


def _build_stride2_downsample(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    out_channels: int,
    in_tensor_type: int,
    weight_tensor_type: int,
) -> onnx.ModelProto:
    """Build STRIDE2_DOWNSAMPLE: Conv(s=2)->Relu->Conv + Conv(1x1,s=2)->Add.

    Main path: Conv1(stride=2,3x3) -> Relu -> Conv2(stride=1,3x3)
    Downsample path: Conv3(stride=2,1x1)
    Both paths produce [batch, out_channels, H/2, W/2] -> Add.
    """
    c_in = in_channels
    c_out = out_channels
    out_h = height // 2
    out_w = width // 2

    # Weights
    w1 = _make_weight("w1", weight_tensor_type, [c_out, c_in, 3, 3])
    w2 = _make_weight("w2", weight_tensor_type, [c_out, c_out, 3, 3])
    w3 = _make_weight("w3", weight_tensor_type, [c_out, c_in, 1, 1])

    # I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, c_in, height, width])
    out = helper.make_tensor_value_info("add1_out", in_tensor_type, [batch_size, c_out, out_h, out_w])

    # Main path
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        name="Conv1",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )
    relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"], name="Relu1")
    conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "w2"],
        outputs=["conv2_out"],
        name="Conv2",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # Downsample path
    conv3 = helper.make_node(
        "Conv",
        inputs=["input", "w3"],
        outputs=["ds_out"],
        name="Conv3",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[2, 2],
    )

    # Merge
    add1 = helper.make_node(
        "Add",
        inputs=["conv2_out", "ds_out"],
        outputs=["add1_out"],
        name="Add1",
    )

    graph = helper.make_graph(
        nodes=[conv1, relu1, conv2, conv3, add1],
        name="Stride2Downsample",
        inputs=[inp],
        outputs=[out],
        initializer=[w1, w2, w3],
    )

    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Stride-2 residual with 1x1 downsample: Conv(s=2)->Relu->Conv + Conv(1x1,s=2)->Add.",
    )


def _build_frontend(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    out_channels: int,
    in_tensor_type: int,
    weight_tensor_type: int,
) -> onnx.ModelProto:
    """Build FRONTEND: Conv(7x7,s=2)->Relu->MaxPool(s=2).

    Conv spatial: (H + 2*3 - 7) // 2 + 1 = H/2
    MaxPool spatial: (H/2 + 2*1 - 3) // 2 + 1 = H/4
    """
    c_in = in_channels
    c_out = out_channels
    conv_h = height // 2
    conv_w = width // 2
    pool_h = conv_h // 2
    pool_w = conv_w // 2

    # Weights
    w1 = _make_weight("w1", weight_tensor_type, [c_out, c_in, 7, 7])

    # I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, c_in, height, width])
    out = helper.make_tensor_value_info("pool_out", in_tensor_type, [batch_size, c_out, pool_h, pool_w])

    # Nodes
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        name="Conv1",
        kernel_shape=[7, 7],
        pads=[3, 3, 3, 3],
        strides=[2, 2],
    )
    relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"], name="Relu1")
    maxpool1 = helper.make_node(
        "MaxPool",
        inputs=["relu1_out"],
        outputs=["pool_out"],
        name="MaxPool1",
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )

    graph = helper.make_graph(
        nodes=[conv1, relu1, maxpool1],
        name="Frontend",
        inputs=[inp],
        outputs=[out],
        initializer=[w1],
    )

    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Front-end path: Conv(7x7,s=2)->Relu->MaxPool(s=2).",
    )


def _build_dual_residual(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    in_tensor_type: int,
    weight_tensor_type: int,
) -> onnx.ModelProto:
    """Build DUAL_RESIDUAL: Two residual blocks with identity Reshape between.

    Block 1: Conv1->Relu1->Conv2->Add1 (skip from input)
    Reshape: identity reshape (FusionEdge trigger for split_fusion_groups)
    Block 2: Conv3->Relu2->Conv4->Add2 (skip from reshape_out)

    The Reshape shape tensor is an initializer (not a graph input) to avoid
    creating an unwanted InEdge data-flow path.
    """
    c = in_channels

    # Weights (4 conv layers)
    w1 = _make_weight("w1", weight_tensor_type, [c, c, 3, 3])
    w2 = _make_weight("w2", weight_tensor_type, [c, c, 3, 3])
    w3 = _make_weight("w3", weight_tensor_type, [c, c, 3, 3])
    w4 = _make_weight("w4", weight_tensor_type, [c, c, 3, 3])

    # Reshape shape tensor (identity: same shape as input)
    reshape_shape = helper.make_tensor(
        "reshape_shape",
        TensorProto.INT64,
        [4],
        np.array([batch_size, c, height, width], dtype=np.int64),
    )

    # I/O
    inp = helper.make_tensor_value_info("input", in_tensor_type, [batch_size, c, height, width])
    out = helper.make_tensor_value_info("add2_out", in_tensor_type, [batch_size, c, height, width])

    # Block 1
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        name="Conv1",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"], name="Relu1")
    conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "w2"],
        outputs=["conv2_out"],
        name="Conv2",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    add1 = helper.make_node(
        "Add",
        inputs=["conv2_out", "input"],
        outputs=["add1_out"],
        name="Add1",
    )

    # FusionEdge boundary: identity Reshape
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["add1_out", "reshape_shape"],
        outputs=["reshape_out"],
        name="BlockBoundary",
    )

    # Block 2
    conv3 = helper.make_node(
        "Conv",
        inputs=["reshape_out", "w3"],
        outputs=["conv3_out"],
        name="Conv3",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu2 = helper.make_node("Relu", inputs=["conv3_out"], outputs=["relu2_out"], name="Relu2")
    conv4 = helper.make_node(
        "Conv",
        inputs=["relu2_out", "w4"],
        outputs=["conv4_out"],
        name="Conv4",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    add2 = helper.make_node(
        "Add",
        inputs=["conv4_out", "reshape_out"],
        outputs=["add2_out"],
        name="Add2",
    )

    graph = helper.make_graph(
        nodes=[conv1, relu1, conv2, add1, reshape_node, conv3, relu2, conv4, add2],
        name="DualResidual",
        inputs=[inp],
        outputs=[out],
        initializer=[w1, w2, w3, w4, reshape_shape],
    )

    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Dual residual blocks with identity Reshape FusionEdge boundary between blocks.",
    )


def make_resnet_subgraph(config: ResNetSubgraphConfig | None = None) -> str:
    """Build an ONNX model for a ResNet18 sub-graph pattern.

    All patterns return just onnx_path (str). The dual-residual pattern embeds
    an identity Reshape node as a FusionEdge trigger -- the split point is
    controlled by the builder, not auto-detected. Mapping is auto-generated
    by GenericMappingGenerator for all patterns.

    Args:
        config: Sub-graph configuration. Uses BASIC_RESIDUAL defaults if None.

    Returns:
        Path to the saved ONNX model.
    """
    if config is None:
        config = ResNetSubgraphConfig(pattern=ResNetPattern.BASIC_RESIDUAL)

    if config.in_dtype not in _DTYPE_MAP or config.weight_dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported data type. Supported types: {list(_DTYPE_MAP.keys())}")

    in_tensor_type = _DTYPE_MAP[config.in_dtype]
    weight_tensor_type = _DTYPE_MAP[config.weight_dtype]

    batch_size = int(config.batch_size)
    in_channels = int(config.in_channels)
    height = int(config.height)
    width = int(config.width)
    out_channels = int(config.out_channels)

    if config.pattern == ResNetPattern.BASIC_RESIDUAL:
        # out_channels must equal in_channels for skip connection
        model = _build_basic_residual(batch_size, in_channels, height, width, in_tensor_type, weight_tensor_type)
        dims_label = f"{batch_size}_{in_channels}_{height}_{width}"

    elif config.pattern == ResNetPattern.STRIDE2_DOWNSAMPLE:
        # Override for valid stride-2 arithmetic
        height, width = 32, 32
        out_channels = 16
        model = _build_stride2_downsample(
            batch_size, in_channels, height, width, out_channels, in_tensor_type, weight_tensor_type
        )
        dims_label = f"{batch_size}_{in_channels}_{height}_{width}"

    elif config.pattern == ResNetPattern.FRONTEND:
        # Override for RGB input + valid stride-2 arithmetic
        in_channels, height, width, out_channels = 3, 32, 32, 8
        model = _build_frontend(
            batch_size, in_channels, height, width, out_channels, in_tensor_type, weight_tensor_type
        )
        dims_label = f"{batch_size}_{in_channels}_{height}_{width}"

    elif config.pattern == ResNetPattern.DUAL_RESIDUAL:
        # out_channels must equal in_channels for skip connections
        model = _build_dual_residual(batch_size, in_channels, height, width, in_tensor_type, weight_tensor_type)
        dims_label = f"{batch_size}_{in_channels}_{height}_{width}"

    else:
        raise ValueError(f"Unknown pattern: {config.pattern}")

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    pattern_name = config.pattern.name.lower()
    onnx_path = os.path.join(
        os.path.dirname(__file__),
        f"resnet_{pattern_name}_{dims_label}.onnx",
    )
    onnx.save(inferred, onnx_path)
    print(f"ResNet {config.pattern.name} ONNX model created: {onnx_path}")

    return onnx_path
