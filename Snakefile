configfile: "workflow/config/params.yaml"
include: "workflow/rules/gemm.smk"

GEMM = config["gemm"]
# PROFILES = ["single_core",]
# PROFILES = ["single_col",]
# PROFILES = ["whole_array",]
# PROFILES = ["single_core", "single_col"]
PROFILES = ["single_core", "single_col", "whole_array"]
# PROFILES = ["single_core", "whole_array"]


# helper: build targets for one profile with zipped M,K,N
def profile_targets(profile):
    p = GEMM[profile]
    return expand(
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/status.ok",
        stream_hw_id=p["stream_hw_id"],
        M=p["Ms"],
        K=p["Ks"],
        N=p["Ns"],
    )

# all targets = concat of both profiles’ zipped combos
def all_targets():
    t = []
    for prof in PROFILES:
        t += profile_targets(prof)
    return t

rule all:
    input:
        all_targets()
