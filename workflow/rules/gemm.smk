import os
from workflow.utils.helpers import get_suffix, get_job_by_suffix

# Rule 1: Generate output.mlir file from main script
rule run_stream_aie_to_generate_mlir_output:
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/output.mlir"
    shell:
        "python3 main_aie_codegen_gemm.py --M {wildcards.M} --K {wildcards.K} --N {wildcards.N}"

# Rule 2: Canonicalize and copy the MLIR into mlir-aie build dir
rule copy_stream_mlir_output_to_mlir_aie:
    input:
        rules.run_stream_aie_to_generate_mlir_output.output
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}.mlir",
        # "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/build/aie_trace_{M}x{K}x{N}_32x32x32.mlir",

    shell:
        # if "stream" in output[0]:
        "aie-opt --canonicalize {input[0]} -o {output[0]} && "
        "echo '✅ Canonicalized MLIR copied to mlir-aie build directory.'"
        # else:
            # print("Skipping MLIR copy because USE_STREAM_OUTPUT is false.")

# Rule 3: Run trace using the copied MLIR
rule run_trace:
    input:
        rules.copy_stream_mlir_output_to_mlir_aie.output,
    output:
        "mlir-aie/programming_examples/basic/matrix_multiplication_stream/{stream_hw_id}/trace_mm_{M}_{K}_{N}.json"
    run:
        shell(
            "set +u && "  # undo standard behavior that uses 'set -u' which doesn't work with env_setup.sh of mlir-aie
            "source mlir-aie/utils/env_setup.sh && "
            "cd mlir-aie/programming_examples/basic/matrix_multiplication_stream/{wildcards.stream_hw_id} && "
            "make trace M={wildcards.M} K={wildcards.K} N={wildcards.N}"
        )

# Rule 4: Post-process the trace
rule postprocess_trace:
    input:
        rules.run_trace.output
    output:
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/trace_efficiency_mm.json",
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/trace_efficiency_mm.png"
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
            --n 32
        """