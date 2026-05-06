import copy
import os

import yaml


def make_swiglu_mapping(
    seq_len, embedding_dim, hidden_dim, last_gemm_down, seq_len_tile_size, embedding_tile_size, hidden_tile_size
):  # noqa: N803
    """
    This mapping assumes that m rows are computed for each Gemm in a pipelined fashion.
    Each layer is partitioned across four rows of compute tiles with inter_core_tiling in the m dimension
    It also assumes that line_size columns are computed in a pipelined fashion."""
    name = f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}_v2.yaml")

    SEQ_LEN_TILE_SIZE = seq_len_tile_size
    INPUT_CHANNEL_TILE_SIZE = embedding_tile_size
    OUTPUT_CHANNEL_TILE_SIZE = hidden_tile_size
    assert seq_len % 4 == 0, "seq_len must be divisible by 4 for this mapping"
    assert seq_len >= SEQ_LEN_TILE_SIZE * 4, f"seq_len must be at least {SEQ_LEN_TILE_SIZE * 4} for this mapping"
    assert embedding_dim % INPUT_CHANNEL_TILE_SIZE == 0, (
        f"embedding_dim must be divisible by {INPUT_CHANNEL_TILE_SIZE} for this mapping"
    )
    assert hidden_dim % OUTPUT_CHANNEL_TILE_SIZE == 0, (
        f"hidden_dim must be divisible by {OUTPUT_CHANNEL_TILE_SIZE} for this mapping"
    )

    # Left Gemm
    if seq_len_tile_size == 1:
        kernel_gemm = {"name": "matvec", "kwargs": {"utilization": 61.8, "layout": "default"}}
    else:
        kernel_gemm = {
            "name": "gemm",
            "kwargs": {
                "utilization": 61.8,
                "m": seq_len_tile_size,
                "k": INPUT_CHANNEL_TILE_SIZE,
                "n": OUTPUT_CHANNEL_TILE_SIZE,
                "layout": "default",
            },
        }
    inter_core_tiling_gemm_left = [
        [{"dim": "D0", "split": 4}, {"dim": "D2", "split": 2}],
    ]
    compute_allocation_gemm_left = [
        [2, 3, 4, 5, 8, 9, 10, 11],
    ]
    # inter_core_tiling_gemm_left = [{"dim": "D2", "split": 2}]
    # compute_allocation_gemm_left = [2, 8]
    gemm_left = {
        "name": "Gemm_Left",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_left),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_left),
        "kernel": copy.deepcopy(kernel_gemm),
    }
    # Right Gemm
    inter_core_tiling_gemm_right = [
        [{"dim": "D0", "split": 4}, {"dim": "D2", "split": 2}],
    ]
    compute_allocation_gemm_right = [
        [14, 15, 16, 17, 20, 21, 22, 23],
    ]
    # inter_core_tiling_gemm_right = [{"dim": "D2", "split": 2}]
    # compute_allocation_gemm_right = [14, 20]
    gemm_right = {
        "name": "Gemm_Right",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_right),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_right),
        "kernel": copy.deepcopy(kernel_gemm),
    }
    # SiLU specific mapping entries. SiLU uses SIMDParser which for two dims goes to (B, H)
    compute_allocation_silu = [
        [26, 27, 28, 29],
    ]
    inter_core_tiling_silu = [
        [{"dim": "D0", "split": 4}],
    ]
    # compute_allocation_silu = [26]
    # inter_core_tiling_silu = []
    kernel_silu = {"name": "silu", "kwargs": {"utilization": 50.0, "layout": "default"}}  # TODO: utilization
    silu = {
        "name": "Silu",
        "core_allocation": copy.deepcopy(compute_allocation_silu),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_silu),
        "kernel": copy.deepcopy(kernel_silu),
    }

    # Elementwise Mul specific mapping entries
    compute_allocation_mul = [
        [32, 33, 34, 35],
    ]
    inter_core_tiling_mul = [
        [{"dim": "D0", "split": 4}],
    ]
    # compute_allocation_mul = [32]
    # inter_core_tiling_mul = []
    kernel_mul = {"name": "eltwise_mul", "kwargs": {"utilization": 50.0, "layout": "default"}}  # TODO: utilization
    mul = {
        "name": "Elt_Mul",
        "core_allocation": copy.deepcopy(compute_allocation_mul),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_mul),
        "kernel": copy.deepcopy(kernel_mul),
    }

    # Final down projection Gemm
    if last_gemm_down:
        inter_core_tiling_gemm_down = [
            [{"dim": "D0", "split": 4}, {"dim": "D2", "split": 2}],
        ]
        compute_allocation_gemm_down = [
            [38, 39, 40, 41, 44, 45, 46, 47],
        ]
        # inter_core_tiling_gemm_down = [{"dim": "D2", "split": 2}]
        # compute_allocation_gemm_down = [38, 44]
        kernel_gemm = {
            "name": "gemm",
            "kwargs": {
                "utilization": 61.8,
                "m": seq_len_tile_size,
                "k": OUTPUT_CHANNEL_TILE_SIZE,
                "n": INPUT_CHANNEL_TILE_SIZE,
                "layout": "default",
            },
        }
        gemm_down = {
            "name": "Gemm_Down",
            "core_allocation": copy.deepcopy(compute_allocation_gemm_down),
            "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_down),
            "kernel": copy.deepcopy(kernel_gemm),
        }
        layers = [gemm_left, gemm_right, silu, mul, gemm_down]
        runtime_args = {
            "input": {},
            "weights_1": {},
            "weights_2": {},
            "weights_3": {},
            "output": {},
        }
    else:
        layers = [gemm_left, gemm_right, silu, mul]
        runtime_args = {
            "input": {},
            "weights_1": {},
            "weights_2": {},
            "output": {},
        }

    # Fused groups; Only one group of all operators with Gemm_Left.D0 dimension
    fused_groups = {
        "name": "Fused_Group_1",
        "layers": [layer["name"] for layer in layers],
        "intra_core_tiling": [
            {"dim": "Gemm_Left.D1", "tile": INPUT_CHANNEL_TILE_SIZE},
            {"dim": "Gemm_Left.D2", "tile": OUTPUT_CHANNEL_TILE_SIZE},
            {"dim": "Gemm_Left.D0", "tile": SEQ_LEN_TILE_SIZE},
        ],
    }
    if last_gemm_down:
        fused_groups["intra_core_tiling"].insert(1, {"dim": "Gemm_Down.D2", "tile": INPUT_CHANNEL_TILE_SIZE})
    mapping = {
        "layers": layers,
        "fused_groups": [fused_groups],
        "runtime_args": runtime_args,
    }

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU mapping file created: {output_file}")
    return output_file
