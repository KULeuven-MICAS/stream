import numpy as np
import onnx
import onnx.shape_inference
import yaml
from onnx import TensorProto, helper


def make_gemm_mapping_single_core(M, K, N, m, k, n, has_mem_tile: bool = False):  # noqa: N803
    name = f"gemm_{M}_{K}_{N}"
    output_file = f"stream/inputs/aie/mapping/{name}.yaml"
    # Construct tiling entries as comma-separated strings
    intra_core_tiling = [
        f"C, {K // k}",
        f"D, {M // m}",
        f"K, {N // n}",
    ]

    inter_core_tiling = ["K, 1"]
    compute_allocation = [1] if not has_mem_tile else [2]
    kernel = {"name": f"mm_{m}x{k}x{n}", "utilization": 61.8}
    mapping = [
        {
            "name": "Gemm",
            "core_allocation": compute_allocation,
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
        {
            "name": "default",
            "core_allocation": compute_allocation,
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
    ]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    return output_file


def make_gemm_mapping_single_col(M, K, N, m, k, n, has_mem_tile: bool = False, nb_compute_cores: int = 4):  # noqa: N803
    name = f"gemm_{M}_{K}_{N}"
    output_file = f"stream/inputs/aie/mapping/{name}_col.yaml"
    # Construct tiling entries as comma-separated strings
    k_inter_core = min(N // n, nb_compute_cores)
    k_intra_core = N // n // k_inter_core
    intra_core_tiling = [
        f"K, {k_intra_core}",
        f"C, {K // k}",
        f"D, {M // m}",
    ]
    inter_core_tiling = [f"K, {k_inter_core}"]
    compute_allocation = (
        [i + 1 for i in range(k_inter_core)] if not has_mem_tile else [i + 2 for i in range(k_inter_core)]
    )
    kernel = {"name": f"mm_{m}x{k}x{n}", "utilization": 61.8}
    mapping = [
        {
            "name": "Gemm",
            "core_allocation": compute_allocation,
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
        {
            "name": "default",
            "core_allocation": compute_allocation,
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
    ]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    return output_file


def make_gemm_workload(M, K, N, in_dtype, out_dtype):  # noqa: N803
    if "16" in in_dtype:
        ACT_SIZE = 16
        WEIGHT_SIZE = 16
    elif "32" in in_dtype:
        ACT_SIZE = 32
        WEIGHT_SIZE = 32
    else:
        raise ValueError(f"Unsupported input data type: {in_dtype}")
    if "16" in out_dtype:
        OUTPUT_SIZE = 16
    elif "32" in out_dtype:
        OUTPUT_SIZE = 32
    else:
        raise ValueError(f"Unsupported output data type: {out_dtype}")

    name = f"gemm_{M}_{K}_{N}"

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
    model = make_gemm_workload(M, N, K)
    onnx.save(model, f"gemm_{M}_{K}_{N}.onnx")
    print(f"Model exported to gemm_{M}_{K}_{N}.onnx with shape inference.")
