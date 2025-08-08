import yaml


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
