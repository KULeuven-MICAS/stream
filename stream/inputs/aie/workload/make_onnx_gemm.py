import numpy as np
import onnx
import onnx.helper as helper
import onnx.shape_inference
from onnx import TensorProto


def make_gemm(M, N, K):
    ACT_SIZE = 16
    WEIGHT_SIZE = 16
    OUTPUT_SIZE = 16

    name = f"gemm_{M}_{N}_{K}"

    # Define the model's graph
    input_tensor_A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, None
    )  # Let shape inference infer the output shape

    # Gemm node
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["A", "B"],
        outputs=["Y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
        act_size=ACT_SIZE,
        weight_size=WEIGHT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    # Omit the weight values by creating dummy weight initializers, just for shape definition
    weight_B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [K, N],
        np.zeros((K, N)),
    )
    weight_B.ClearField("float_data")

    # Create the graph and model
    graph = helper.make_graph(
        nodes=[gemm_node],
        name=name,
        inputs=[input_tensor_A],
        outputs=[output_tensor],
        initializer=[weight_B],  # Include only the shapes of the weights, not the values
    )

    # Create the model
    model = helper.make_model(graph, producer_name="stream-aie")

    # Run shape inference to automatically infer the shapes of all tensors
    inferred_model = onnx.shape_inference.infer_shapes(model)

    # Save the model to file
    save_path = f"stream/inputs/aie/workload/{name}.onnx"
    onnx.save(inferred_model, save_path)

    print(f"{name} exported to {save_path}.")

    return save_path


if __name__ == "__main__":
    # Example usage
    M = 128
    N = 128
    K = 64
    model = make_gemm(M, N, K)
    onnx.save(model, f"gemm_{M}_{N}_{K}.onnx")
    print(f"Model exported to gemm_{M}_{N}_{K}.onnx with shape inference.")
