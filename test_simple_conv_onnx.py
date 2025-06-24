import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

# Model parameters
batch_size = 1
in_channels = 8
out_channels = 8
height = width = 56
kernel_size = 3
stride = 1
padding = 1

# Input tensor
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [batch_size, in_channels, height, width])
output_tensor = helper.make_tensor_value_info(
    "output", onnx.TensorProto.FLOAT, [batch_size, out_channels, height, width]
)

# Conv1 weights (no bias)
w1 = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
w1_initializer = numpy_helper.from_array(w1, name="conv1_weight")

# Conv2 weights (no bias)
w2 = np.random.randn(out_channels, out_channels, kernel_size, kernel_size).astype(np.float32)
w2_initializer = numpy_helper.from_array(w2, name="conv2_weight")

# Conv1 node (no bias input)
conv1_node = helper.make_node(
    "Conv",
    name="conv1",
    inputs=["input", "conv1_weight"],
    outputs=["conv1_out"],
    kernel_shape=[kernel_size, kernel_size],
    strides=[stride, stride],
    pads=[padding, padding, padding, padding],
)

# Conv2 node (no bias input)
conv2_node = helper.make_node(
    "Conv",
    name="conv2",
    inputs=["conv1_out", "conv2_weight"],
    outputs=["output"],
    kernel_shape=[kernel_size, kernel_size],
    strides=[stride, stride],
    pads=[padding, padding, padding, padding],
)

# Graph
graph = helper.make_graph(
    [conv1_node, conv2_node], "SimpleConvNet", [input_tensor], [output_tensor], [w1_initializer, w2_initializer]
)

# Model
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

# Save with external weights
onnx_file = "simple_conv.onnx"
external_weights_file = "simple_conv_data"
onnx.save_model(
    model, onnx_file, save_as_external_data=True, all_tensors_to_one_file=True, location=external_weights_file
)

# Shape inference
onnx_model = onnx.load(onnx_file)
inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.save(inferred_model, onnx_file)

print(f"ONNX model saved to {onnx_file} with external weights in {external_weights_file}")
