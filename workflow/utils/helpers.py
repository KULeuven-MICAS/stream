def shape_id(shape):
    """Create a unique identifier from shape parameters."""
    return f"M{shape['M']}_N{shape['N']}_K{shape['K']}"

def get_suffix(cfg):
    return f"{cfg['M']}x{cfg['K']}x{cfg['N']}_{cfg['m']}x{cfg['k']}x{cfg['n']}_{'stream' if cfg['use_stream_output'] else 'nostream'}"

def get_job_by_suffix(suffix, jobs):
    for cfg in jobs:
        if get_suffix(cfg) == suffix:
            cfg["suffix"] = suffix
            return cfg
    raise ValueError(f"No job found for suffix '{suffix}'")

def get_mlir_output_path(cfg):
    return f"outputs/{cfg['stream_hw_identifier']}-gemm_{cfg['M']}_{cfg['K']}_{cfg['N']}-fused-constraint-optimization/output.mlir"
