import os

# Rule 1: Generate output.mlir file from main script
rule run_stream_aie_to_generate_mlir_output:
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/output.mlir"
    # log:
    #     "logs/gemm_{stream_hw_id}_{M}_{K}_{N}.log"
    shell:
        """
        python3 main_aie_codegen_gemm_mem_tile.py --M {wildcards.M} --K {wildcards.K} --N {wildcards.N} | tee {log}
        """

# Rule 2: Canonicalize and copy the MLIR into mlir-aie build dir
rule copy_stream_mlir_output_to_mlir_aie:
    input:
        rules.run_stream_aie_to_generate_mlir_output.output
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}.mlir",
        # "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}_32x32x32.mlir",
    shell:
        """
        aie-opt --canonicalize {input[0]} -o {output[0]} && \
        echo '✅ Canonicalized MLIR copied to mlir-aie build directory.' \
        """

# Rule 3: Run trace using the copied MLIR
rule run_trace:
    input:
        rules.copy_stream_mlir_output_to_mlir_aie.output,
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/trace_mm_{M}_{K}_{N}.json"
    log:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/run_trace.log"
    shell:
        """
        (
            set +u && \
            source mlir-aie/utils/env_setup.sh && \
            cd mlir-aie/programming_examples/basic/matrix_multiplication_stream/{wildcards.stream_hw_id} && \
            make trace M={wildcards.M} K={wildcards.K} N={wildcards.N} \
        ) > {log} 2>&1
        """

# Rule 4: Post-process the trace
rule postprocess_trace:
    input:
        rules.run_trace.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/trace_efficiency_mm.json",
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/trace_efficiency_mm.png"
    # log:
    #     "logs/gemm_{stream_hw_id}_{M}_{K}_{N}.log"
    shell:
        """
        python3 postprocess_aie_trace.py \
            --input {input[0]} \
            --output {output[0]} \
            --fig {output[1]} \
            --M {wildcards.M} \
            --K {wildcards.K} \
            --N {wildcards.N} \
            --m 32 \
            --k 32 \
            --n 32 \
        """

# Rule 5: Create a status file to indicate successful completion
rule mark_success:
    input:
        rules.postprocess_trace.output[0]  # assumes trace_efficiency_mm.json
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/status.ok"
    shell:
        "echo 'success' > {output}"