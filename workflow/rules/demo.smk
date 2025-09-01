# Demo rules
# Run a fixed single-core demo at M=K=N=128 (no iteration)
rule single_core_demo:
    input:
        # Produces the exact target your gemm rules generate for single_core
        lambda wc: (
            f"outputs/{GEMM['single_core']['stream_hw_id']}"
            f"-gemm_128_128_128-"
            f"{GEMM['single_core']['nb_rows']}_row_{GEMM['single_core']['nb_cols']}_col/status.ok"
        )
# Run a fixed single-col demo at M=K=N=128 (no iteration)
rule single_col_demo:
    input:
        # Produces the exact target your gemm rules generate for single_col
        lambda wc: (
            f"outputs/{GEMM['single_col']['stream_hw_id']}"
            f"-gemm_128_128_128-"
            f"{GEMM['single_col']['nb_rows']}_row_{GEMM['single_col']['nb_cols']}_col/status.ok"
        )