import numpy as np
import onnx
import onnx.helper as helper
import onnx.shape_inference
from onnx import TensorProto

# Define the model's graph
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 112, 112])
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, None
)  # Let shape inference infer the output shape

# First convolution: 3 input channels, 16 output channels, 3x3 kernel, stride 2, padding 1
conv1_node = helper.make_node(
    "Conv",
    inputs=["input", "conv1_weights"],  # No bias term
    outputs=["conv1_output"],
    kernel_shape=[1, 1],
    pads=[0, 0, 0, 0],  # Padding for height and width
    strides=[1, 1],  # Stride for height and width
)

# Second convolution: 16 input channels, 32 output channels, 3x3 kernel, stride 2, padding 1
conv2_node = helper.make_node(
    "Conv",
    inputs=["conv1_output", "conv2_weights"],  # No bias term
    outputs=["output"],
    kernel_shape=[1, 1],
    pads=[0, 0, 0, 0],  # Padding for height and width
    strides=[1, 1],  # Stride for height and width
)

# Omit the weight values by creating dummy weight initializers, just for shape definition
conv1_weights = helper.make_tensor(
    "conv1_weights",
    TensorProto.FLOAT,
    [8, 3, 1, 1],  # 8 output channels, 3 input channels, 3x3 kernel
    np.zeros((8, 3, 1, 1)),
)
conv1_weights.ClearField("float_data")

conv2_weights = helper.make_tensor(
    "conv2_weights",
    TensorProto.FLOAT,
    [16, 8, 1, 1],  # 16 output channels, 8 input channels, 3x3 kernel
    np.zeros((16, 8, 1, 1)),
)
conv2_weights.ClearField("float_data")

# Create the graph and model
graph = helper.make_graph(
    nodes=[conv1_node, conv2_node],
    name="conv1x1_conv1x1",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[conv1_weights, conv2_weights],  # Include only the shapes of the weights, not the values
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-example")

# Run shape inference to automatically infer the shapes of all tensors
inferred_model = onnx.shape_inference.infer_shapes(model)

# Save the model to file
onnx.save(inferred_model, "conv1x1_conv1x1.onnx")

print("Model exported to conv1x1_conv1x1.onnx with shape inference.")
