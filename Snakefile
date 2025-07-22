configfile: "workflow/config/params.yaml"

include: "workflow/rules/gemm.smk"

from workflow.utils.helpers import shape_id, get_suffix

defaults = config["gemm"]["defaults"]
shapes = config["gemm"]["shapes"]
stream_hw_id = defaults["stream_hw_identifier"]

rule all:
    input:
        expand(
            "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/trace_efficiency_mm.json",
            stream_hw_id=stream_hw_id,
            M=[shape["M"] for shape in shapes],
            K=[shape["K"] for shape in shapes],
            N=[shape["N"] for shape in shapes],
        )