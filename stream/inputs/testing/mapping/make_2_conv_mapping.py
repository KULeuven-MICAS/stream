import copy
import os

import yaml

from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig


def make_2_conv_mapping(config: TwoConvWorkloadConfig):  # noqa: N803
    """
    This mapping assumes that m rows are computed for each Gemm in a pipelined fashion.
    Each layer is partitioned across four rows of compute tiles with inter_core_tiling in the m dimension
    It also assumes that line_size columns are computed in a pipelined fashion."""
    name = (
        f"2conv_{config.batch_size}_{config.height}_{config.width}"
        f"_{config.in_channels}_{config.out_channels_1}_{config.out_channels_2}_{config.kernel_size}"
    )
    output_file = os.path.join(os.path.dirname(__file__), f"{name}.yaml")

    TILE_DIM = "Conv1.D0"  # Tiling dimension (batch dimension)
    TILE_SIZE = 1  # Tiling size per steady state iteration
    assert config.out_channels_1 % 4 == 0, "out_channels_1 must be divisible by 4 for this mapping"
    assert config.out_channels_2 % 4 == 0, "out_channels_2 must be divisible by 4 for this mapping"

    # Left Gemm
    inter_core_tiling_conv1 = [{"dim": "D6", "split": 4}]
    kernel_conv1 = {"name": "conv", "kwargs": {}}
    compute_allocation_conv1 = [0, 1, 2, 3]
    conv1 = {
        "name": "Conv1",
        "core_allocation": [copy.deepcopy(compute_allocation_conv1)],
        "inter_core_tiling": [copy.deepcopy(inter_core_tiling_conv1)],
        "kernel": copy.deepcopy(kernel_conv1),
    }
    # Right Gemm
    inter_core_tiling_conv2 = [{"dim": "D6", "split": 4}]
    kernel_conv2 = {"name": "conv", "kwargs": {}}
    compute_allocation_conv2 = [0, 1, 2, 3]
    conv2 = {
        "name": "Conv2",
        "core_allocation": [copy.deepcopy(compute_allocation_conv2)],
        "inter_core_tiling": [copy.deepcopy(inter_core_tiling_conv2)],
        "kernel": copy.deepcopy(kernel_conv2),
    }

    layers = [conv1, conv2]
    runtime_args = {
        "input": {},
        "weights_1": {},
        "weights_2": {},
        "output": {},
    }

    # Fused groups; Only one group of all operators with Gemm_Left.D0 dimension
    fused_groups = {
        "name": "Fused_Group_1",
        "layers": [layer["name"] for layer in layers],
        "intra_core_tiling": [
            {"dim": TILE_DIM, "tile": TILE_SIZE},
        ],
    }

    mapping = {
        "layers": layers,
        "fused_groups": [fused_groups],
        "runtime_args": runtime_args,
    }

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU mapping file created: {output_file}")
    return output_file
