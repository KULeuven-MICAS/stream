import copy
import os

import yaml


def make_swiglu_mapping_pipelined(seq_len, embedding_dim, hidden_dim, m, k, n, line_size):  # noqa: N803
    """
    This mapping assumes that m rows are computed for each Gemm in a pipelined fashion.
    It also assumes that line_size columns are computed in a pipelined fashion."""
    name = f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}.yaml")

    # General mapping entries for all operators
    inter_core_tiling = ["K, 1"]
    compute_allocation_gemm_left = [2]
    compute_allocation_gemm_right = [3]

    # Left and right Gemms specific mapping entries
    intra_core_tiling_gemm = [
        f"C, {embedding_dim // k}",
        f"K, {hidden_dim // n}",
        f"D, {seq_len}",
    ]
    kernel_gemm = {"name": "matvec", "kwargs": {"utilization": 61.8}}
    gemm_left = {
        "name": "Gemm_Left",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_left),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_gemm),
    }
    gemm_right = {
        "name": "Gemm_Right",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_right),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    # SiLU specific mapping entries. SiLU uses SIMDParser which for two dims goes to (B, H)
    compute_allocation_silu = [4]
    intra_core_tiling_silu = [
        f"H, {hidden_dim // line_size}",
        f"B, {seq_len // 1}",
    ]
    kernel_silu = {"name": "silu", "kwargs": {"utilization": 50.0}}  # TODO: utilization
    silu = {
        "name": "Silu",
        "core_allocation": copy.deepcopy(compute_allocation_silu),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_silu),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_silu),
    }

    # Elementwise Mul specific mapping entries
    compute_allocation_mul = [5]
    intra_core_tiling_mul = [
        f"H, {hidden_dim // line_size}",
        f"B, {seq_len // 1}",
    ]
    kernel_mul = {"name": "eltwise_mul", "kwargs": {"utilization": 50.0}}  # TODO: utilization
    mul = {
        "name": "Elt_Mul",
        "core_allocation": copy.deepcopy(compute_allocation_mul),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_mul),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_mul),
    }

    # Final down projection Gemm
    compute_allocation_gemm_down = [11]
    gemm_down = {
        "name": "Gemm_Down",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_down),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    # Default specific mapping entries
    default = {
        "name": "default",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_left),
        "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    mapping = [gemm_left, gemm_right, silu, mul, gemm_down, default]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU mapping file created: {output_file}")
    return output_file


def make_swiglu_mapping_pipelined2(seq_len, embedding_dim, hidden_dim, m, k, n, line_size):  # noqa: N803
    """
    This mapping assumes that m rows are computed for each Gemm in a pipelined fashion.
    Each layer is partitioned across four rows of compute tiles with inter_core_tiling in the m dimension
    It also assumes that line_size columns are computed in a pipelined fashion."""
    name = f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}_v2.yaml")

    assert seq_len % 4 == 0, "seq_len must be divisible by 4 for this mapping"

    # Left and right Gemms specific mapping entries
    inter_core_tiling_gemm = ["D0, 4"]
    # intra_core_tiling_gemm = [
    #     f"C, {embedding_dim // k}",
    #     f"K, {hidden_dim // n}",
    #     f"D, {seq_len // 4}",
    # ]
    kernel_gemm = {"name": "matvec", "kwargs": {"utilization": 61.8}}
    compute_allocation_gemm_left = [2, 3, 4, 5]
    gemm_left = {
        "name": "Gemm_Left",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_left),
        # "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm),
        "kernel": copy.deepcopy(kernel_gemm),
    }
    compute_allocation_gemm_right = [8, 9, 10, 11]
    gemm_right = {
        "name": "Gemm_Right",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_right),
        # "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    # SiLU specific mapping entries. SiLU uses SIMDParser which for two dims goes to (B, H)
    compute_allocation_silu = [14, 15, 16, 17]
    inter_core_tiling_silu = ["D0, 4"]
    # intra_core_tiling_silu = [
    #     f"H, {hidden_dim // line_size}",
    #     f"B, {seq_len // 4}",
    # ]
    kernel_silu = {"name": "silu", "kwargs": {"utilization": 50.0}}  # TODO: utilization
    silu = {
        "name": "Silu",
        "core_allocation": copy.deepcopy(compute_allocation_silu),
        # "intra_core_tiling": copy.deepcopy(intra_core_tiling_silu),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_silu),
        "kernel": copy.deepcopy(kernel_silu),
    }

    # Elementwise Mul specific mapping entries
    compute_allocation_mul = [20, 21, 22, 23]
    inter_core_tiling_mul = ["D0, 4"]
    # intra_core_tiling_mul = [
    #     f"H, {hidden_dim // line_size}",
    #     f"B, {seq_len // 4}",
    # ]
    kernel_mul = {"name": "eltwise_mul", "kwargs": {"utilization": 50.0}}  # TODO: utilization
    mul = {
        "name": "Elt_Mul",
        "core_allocation": copy.deepcopy(compute_allocation_mul),
        # "intra_core_tiling": copy.deepcopy(intra_core_tiling_mul),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_mul),
        "kernel": copy.deepcopy(kernel_mul),
    }

    # Final down projection Gemm
    compute_allocation_gemm_down = [26, 27, 28, 29]
    gemm_down = {
        "name": "Gemm_Down",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_down),
        # "intra_core_tiling": copy.deepcopy(intra_core_tiling_gemm),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    mapping = [gemm_left, gemm_right, silu, mul, gemm_down]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU mapping file created: {output_file}")
    return output_file
