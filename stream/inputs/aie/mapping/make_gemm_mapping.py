import os

import yaml

def make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use: int = 4, nb_cols_to_use: int = 4):  # noqa: N803
    NB_COMPUTE_ROWS_OF_ARRAY = 4
    name = f"gemm_{M}_{K}_{N}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}_whole_array.yaml")
    k_inter_core = min(N // n, nb_cols_to_use)
    d_inter_core = min(M // m, nb_rows_to_use)
    compute_allocation = [
        row_idx + 2 * (col_idx + 1) + NB_COMPUTE_ROWS_OF_ARRAY * col_idx
        for col_idx in range(nb_cols_to_use)
        for row_idx in range(nb_rows_to_use)
    ]

    kernel = {
        "name": "gemm",
        "kwargs": {"m": m, "k": k, "n": n, "utilization": 61.8, "layout": "default"},
    }
    inter_core_tiling = [
        {"dim": "D0", "split": d_inter_core}, 
        {"dim": "D2", "split": k_inter_core}
    ]
    gemm = {
        "name": "Gemm",
        "core_allocation": [compute_allocation],
        "inter_core_tiling": [inter_core_tiling],
        "kernel": kernel,
    }
    # Fused groups; Only one group of all operators with Gemm_Left.D0 dimension
    fused_groups = {
        "name": "Fused_Group_1",
        "layers": ["Gemm"],
        "intra_core_tiling": [
            {"dim": "Gemm.D1", "tile": k},
            {"dim": "Gemm.D0", "tile": m},
            {"dim": "Gemm.D2", "tile": n},
        ],
    }
    runtime_args = {
        "A": {},
        "B": {},
        "Y": {},
    }
    mapping = {
        "layers": [gemm],
        "fused_groups": [fused_groups],
        "runtime_args": runtime_args
    }

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"Gemm mapping file created: {output_file}")
    return output_file
