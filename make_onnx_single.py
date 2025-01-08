import numpy as np
import onnx
import onnx.helper as helper
import onnx.shape_inference
from onnx import TensorProto

IC = 64
OC = 64
H = 32
W = 32

name = f"conv1x1_{IC}_{OC}_{H}_{W}"

# Define the model's graph
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, IC, H, W])
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, None
)  # Let shape inference infer the output shape

# First convolution: 3 input channels, 16 output channels, 1x1 kernel, stride 2, padding 1
conv1_node = helper.make_node(
    "Conv",
    inputs=["input", "conv1_weights"],  # No bias term
    outputs=["output"],
    kernel_shape=[1, 1],
    pads=[0, 0, 0, 0],  # Padding for height and width
    strides=[1, 1],  # Stride for height and width
)

# Omit the weight values by creating dummy weight initializers, just for shape definition
conv1_weights = helper.make_tensor(
    "conv1_weights",
    TensorProto.FLOAT,
    [IC, OC, 1, 1],  # 8 output channels, 3 input channels, 1x1 kernel
    np.zeros((IC, OC, 1, 1)),
)
conv1_weights.ClearField("float_data")


# Create the graph and model
graph = helper.make_graph(
    nodes=[conv1_node],
    name=name,
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[conv1_weights],  # Include only the shapes of the weights, not the values
)

# Create the model
model = helper.make_model(graph, producer_name="stream-aie")

# Run shape inference to automatically infer the shapes of all tensors
inferred_model = onnx.shape_inference.infer_shapes(model)

# Save the model to file
onnx.save(inferred_model, f"{name}.onnx")

print(f"Model exported to {name}.onnx with shape inference.")
