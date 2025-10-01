# Demo rules
# Run a fixed single-core demo at M=K=N=128 (no iteration)
rule single_core_demo_npu1:
    input:
        # Produces the exact target your gemm rules generate for single_core_npu1
        lambda wc: (
            f"outputs/{GEMM['single_core_npu1']['stream_hw_id']}"
            f"-gemm_128_128_128-"
            f"{GEMM['single_core_npu1']['nb_rows']}_row_{GEMM['single_core_npu1']['nb_cols']}_col/status.ok"
        )
rule single_core_demo_npu2:
    input:
        # Produces the exact target your gemm rules generate for single_core_npu2
        lambda wc: (
            f"outputs/{GEMM['single_core_npu2']['stream_hw_id']}"
            f"-gemm_128_128_128-"
            f"{GEMM['single_core_npu2']['nb_rows']}_row_{GEMM['single_core_npu2']['nb_cols']}_col/status.ok"
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