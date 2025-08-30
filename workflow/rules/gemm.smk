# Build a lookup so rules can fetch per-profile trace_size based on {stream_hw_id}
GEMM = config["gemm"]
TRACE_SIZE = { v["stream_hw_id"]: v["trace_size"] for v in GEMM.values() }
STREAM_MAIN_FILE = {v["stream_hw_id"]: v["stream_main_file"] for v in GEMM.values()}
WARMUP = {v["stream_hw_id"]: v.get("warmup", 1) for v in GEMM.values()}
ITERATIONS = {v["stream_hw_id"]: v.get("iterations", 5) for v in GEMM.values()}

rule run_stream_aie_to_generate_mlir_output:
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/output.mlir"
    params:
        trace_size = lambda wc: TRACE_SIZE[wc.stream_hw_id],
        stream_main_file = lambda wc: STREAM_MAIN_FILE[wc.stream_hw_id]
    shell:
        """
        python3 {params.stream_main_file} \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --rows {wildcards.nb_rows} --cols {wildcards.nb_cols} \
            --trace_size {params.trace_size} | tee {output}.log
        """

rule copy_stream_mlir_output_to_mlir_aie:
    input:
        rules.run_stream_aie_to_generate_mlir_output.output
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}_32x32x32_{nb_rows}_row_{nb_cols}_col.mlir",
    shell:
        """
        make clean -C mlir-aie/programming_examples/basic/matrix_multiplication_stream/{wildcards.stream_hw_id} && \
        mkdir -p $(dirname {output}) && \
        aie-opt --canonicalize {input} -o {output} && \
        echo "🧹 Cleaned {wildcards.stream_hw_id}"
        echo '✅ Canonicalized MLIR copied.'
        """

rule run_trace:
    input:
        rules.copy_stream_mlir_output_to_mlir_aie.output,
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/trace_mm_{M}_{K}_{N}_{nb_rows}_row_{nb_cols}_col.json"
    log:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/run_trace.log"
    params:
        trace_size = lambda wc: TRACE_SIZE[wc.stream_hw_id],
        warmup = lambda wc: WARMUP[wc.stream_hw_id],
        iterations = lambda wc: ITERATIONS[wc.stream_hw_id],
    shell:
        """
        (
            set +u && \
            source mlir-aie/utils/env_setup.sh && \
            cd mlir-aie/programming_examples/basic/matrix_multiplication_stream/{wildcards.stream_hw_id} && \
            make trace M={wildcards.M} K={wildcards.K} N={wildcards.N} nb_rows={wildcards.nb_rows} nb_cols={wildcards.nb_cols} \
            trace_size={params.trace_size} warmup={params.warmup} iterations={params.iterations}
        ) > {log} 2>&1
        """

rule copy_trace_output:
    input:
        rules.run_trace.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/trace_mm_{M}_{K}_{N}.json"
    shell:
        "mkdir -p $(dirname {output}) && cp {input} {output}"

rule postprocess_wallclock_time:
    input:
        rules.run_trace.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/wall_clock_summary.json"
    log:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/postprocess_wallclock_time.log"
    shell:
        """
        (
            python3 postprocess_wallclock_time.py \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --row {wildcards.nb_rows} --col {wildcards.nb_cols} \
            --hwid {wildcards.stream_hw_id} 
        ) > {log} 2>&1
        """

rule postprocess_trace:
    input:
        rules.copy_trace_output.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/aggregate_perf.json",
    log:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/postprocess_trace.log"
    params:
        warmup = lambda wc: WARMUP[wc.stream_hw_id],
        iterations = lambda wc: ITERATIONS[wc.stream_hw_id],
    shell:
        """
        (        
            python3 postprocess_aie_trace.py \
            --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} \
            --m 32 --k 32 --n 32 \
            --hwid {wildcards.stream_hw_id} \
            --row {wildcards.nb_rows} --col {wildcards.nb_cols} \
            --warmup {params.warmup} --iterations {params.iterations}
        ) > {log} 2>&1
        """

rule mark_success:
    input:
        rules.postprocess_trace.output[0],
        rules.postprocess_wallclock_time.output[0]
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/status.ok"
    shell:
        "echo success > {output}"
