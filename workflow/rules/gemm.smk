# Build a lookup so rules can fetch per-profile trace_size based on {stream_hw_id}
GEMM = config["gemm"]
TRACE_SIZE = { v["stream_hw_id"]: v["trace_size"] for v in GEMM.values() }
STREAM_MAIN_FILE = {v["stream_hw_id"]: v["stream_main_file"] for v in GEMM.values()}

rule run_stream_aie_to_generate_mlir_output:
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/output.mlir"
    params:
        trace_size = lambda wc: TRACE_SIZE[wc.stream_hw_id],
        stream_main_file = lambda wc: STREAM_MAIN_FILE[wc.stream_hw_id]
    shell:
        """
        python3 {params.stream_main_file} \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --trace_size {params.trace_size} | tee {output}.log
        """

rule copy_stream_mlir_output_to_mlir_aie:
    input:
        rules.run_stream_aie_to_generate_mlir_output.output
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}_32x32x32.mlir",
    shell:
        """
        aie-opt --canonicalize {input[0]} -o {output[0]} && \
        echo 'Canonicalized MLIR copied.'
        """

rule run_trace:
    input:
        rules.copy_stream_mlir_output_to_mlir_aie.output,
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/trace_mm_{M}_{K}_{N}.json"
    threads: 1
    log:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/run_trace.log"
    params:
        trace_size = lambda wc: TRACE_SIZE[wc.stream_hw_id]
    shell:
        """
        (
            set +u && \
            source mlir-aie/utils/env_setup.sh && \
            cd mlir-aie/programming_examples/basic/matrix_multiplication_stream/{wildcards.stream_hw_id} && \
            make trace M={wildcards.M} K={wildcards.K} N={wildcards.N} trace_size={params.trace_size}
        ) > {log} 2>&1
        """

rule copy_trace_output:
    input:
        rules.run_trace.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/traces/trace_mm_{M}_{K}_{N}.json"
    shell:
        "mkdir -p $(dirname {output}) && cp {input} {output}"

rule postprocess_trace:
    input:
        rules.copy_trace_output.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/traces/tile2,1_report.json",
    shell:
        """
        python3 postprocess_aie_trace.py \
            --input {input} \
            --output $(dirname {output}) \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --m 32 --k 32 --n 32 \
            --hwid {wildcards.stream_hw_id} \
        """

rule mark_success:
    input:
        rules.postprocess_trace.output[0]
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/status.ok"
    shell:
        "echo success > {output}"
