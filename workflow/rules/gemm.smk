# Build a lookup so rules can fetch per-profile trace_size based on {stream_hw_id}
gemm_config = config["gemm"]

rule run_stream_aie_to_generate_mlir_output:
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/output.mlir"
    params:
        main_file = gemm_config["stream_main_file"],
        trace_size = gemm_config["trace_size"],
        npu = gemm_config["npu"],
    shell:
        """
        python3 {params.main_file} \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --rows {wildcards.nb_rows} --cols {wildcards.nb_cols} \
            --trace_size {params.trace_size} --npu {params.npu} | tee {output}.log
        """

rule mark_success:
    input:
        rules.run_stream_aie_to_generate_mlir_output.output[0]
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/status.ok"
    shell:
        "echo success > {output}"
