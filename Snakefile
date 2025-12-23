configfile: "workflow/config/params.yaml"
include: "workflow/rules/gemm.smk"
include: "workflow/rules/demo.smk"

from snakemake.exceptions import WorkflowError

print(config)

def profile_keys():
    """Return configured profile names (config:profiles) or default to ['gemm']."""
    return config.get("profiles", ["gemm"])


def get_profiles():
    """Load profile dictionaries lazily so config is guaranteed to be available."""
    profiles = []
    for key in profile_keys():
        if key not in config:
            raise WorkflowError(f"Config missing profile '{key}' (available: {list(config.keys())})")
        profiles.append(config[key])
    return profiles

# helper: build targets for one profile with zipped M,K,N
def profile_targets(profile):
    return expand(
        "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/status.ok",
        stream_hw_id=profile["stream_hw_id"],
        nb_rows=profile["nb_rows"],
        nb_cols=profile["nb_cols"],
        M=profile["Ms"],
        K=profile["Ks"],
        N=profile["Ns"],
    )

# all targets = concat of both profiles’ zipped combos
def all_targets():
    t = []
    for prof in get_profiles():
        t += profile_targets(prof)
    return t

rule all:
    input:
        all_targets()
