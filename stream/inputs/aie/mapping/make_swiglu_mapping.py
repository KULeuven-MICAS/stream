import os

import yaml


def make_swiglu_mapping_pipelined(input_shape, out_channels, m, k, n, line_size):  # noqa: N803
    """
    This mapping assumes that m rows are computed for each Gemm in a pipelined fashion.
    It also assumes that line_size columns are computed in a pipelined fashion."""
    X, Y = int(input_shape[0]), int(input_shape[1])
    name = f"swiglu_{X}_{Y}_{out_channels}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}.yaml")

    # General mapping entries for all operators
    inter_core_tiling = ["K, 1"]
    compute_allocation_gemm_left = [2]
    compute_allocation_gemm_right = [3]

    # Left and right Gemms specific mapping entries
    intra_core_tiling_gemm = [
        f"C, {X // k}",
        f"K, {out_channels // n}",
        f"D, {Y // m}",
    ]
    kernel_gemm = {"name": f"mm_{m}x{k}x{n}", "utilization": 61.8}
    gemm_left = {
        "name": "Gemm_Left",
        "core_allocation": compute_allocation_gemm_left,
        "intra_core_tiling": intra_core_tiling_gemm,
        "inter_core_tiling": inter_core_tiling,
        "kernel": kernel_gemm,
    }
    gemm_right = {
        "name": "Gemm_Right",
        "core_allocation": compute_allocation_gemm_right,
        "intra_core_tiling": intra_core_tiling_gemm,
        "inter_core_tiling": inter_core_tiling,
        "kernel": kernel_gemm,
    }

    # SiLU specific mapping entries. SiLU uses SIMDParser which for two dims goes to (B, H)
    compute_allocation_silu = [4]
    intra_core_tiling_silu = [
        f"H, {out_channels // line_size}",
        f"B, {Y // 1}",
    ]
    kernel_silu = {"name": "silu_bf16", "utilization": 50.0}  # TODO: utilization
    silu = {
        "name": "Silu",
        "core_allocation": compute_allocation_silu,
        "intra_core_tiling": intra_core_tiling_silu,
        "inter_core_tiling": inter_core_tiling,
        "kernel": kernel_silu,
    }

    # Elementwise Mul specific mapping entries
    compute_allocation_mul = [5]
    intra_core_tiling_mul = [
        f"H, {out_channels // line_size}",
        f"B, {Y // 1}",
    ]
    kernel_mul = {"name": "elemwise_mul_bf16", "utilization": 50.0}  # TODO: utilization
    mul = {
        "name": "Elt_Mul",
        "core_allocation": compute_allocation_mul,
        "intra_core_tiling": intra_core_tiling_mul,
        "inter_core_tiling": inter_core_tiling,
        "kernel": kernel_mul,
    }

    # Default specific mapping entries
    default = {
        "name": "default",
        "core_allocation": compute_allocation_gemm_left,
        "intra_core_tiling": intra_core_tiling_gemm,
        "inter_core_tiling": inter_core_tiling,
        "kernel": kernel_gemm,
    }

    mapping = [gemm_left, gemm_right, silu, mul, default]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU mapping file created: {output_file}")
    return output_file
