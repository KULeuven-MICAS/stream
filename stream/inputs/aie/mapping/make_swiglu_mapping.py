import yaml


def make_swiglu_mapping_pipelined(input_shape, out_channels, m, k, n, line_size):  # noqa: N803
    x, y = int(input_shape[0]), int(input_shape[1])
    name = f"swiglu_{x}_{y}_{out_channels}_{line_size}"
    output_file = f"stream/inputs/aie/mapping/{name}.yaml"
    # Calculate the 
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


def make_gemm_mapping_single_col(M, K, N, m, k, n, has_mem_tile: bool = False, nb_rows: int = 4):  # noqa: N803
    name = f"gemm_{M}_{K}_{N}"
    output_file = f"stream/inputs/aie/mapping/{name}_col.yaml"
    # Construct tiling entries as comma-separated strings
    k_inter_core = min(N // n, nb_rows)
    k_intra_core = N // n // k_inter_core
    intra_core_tiling = [
        f"C, {K // k}",
        f"D, {M // m}",
        f"K, {k_intra_core}",
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


def make_gemm_mapping_whole_array(M, K, N, m, k, n, nb_rows_to_use: int = 4, nb_cols_to_use: int = 4):  # noqa: N803
    NB_COMPUTE_ROWS_OF_ARRAY = 4
    name = f"gemm_{M}_{K}_{N}"
    output_file = f"stream/inputs/aie/mapping/{name}_whole_array.yaml"
    # Construct tiling entries as comma-separated strings
    k_inter_core = min(N // n, nb_cols_to_use)
    k_intra_core = N // n // k_inter_core
    d_inter_core = min(M // m, nb_rows_to_use)
    d_intra_core = M // m // d_inter_core
    intra_core_tiling = [
        f"C, {K // k}",
        f"D, {d_intra_core}",
        f"K, {k_intra_core}",
    ]
    inter_core_tiling = [
        f"D, {d_inter_core}",
        f"K, {k_inter_core}",
    ]
    # compute_allocation = (
    #     [i + 1 for i in range(k_inter_core)] if not has_mem_tile else [i + 2 for i in range(k_inter_core)]
    # )
    compute_allocation = [
        row_idx + 2 * (col_idx + 1) + NB_COMPUTE_ROWS_OF_ARRAY * col_idx
        for col_idx in range(nb_cols_to_use)
        for row_idx in range(nb_rows_to_use)
    ]

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
