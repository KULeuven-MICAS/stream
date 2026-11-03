"""The workload and mapping builders write where they are told, leaving the committed fixtures untouched."""

from pathlib import Path

import pytest

from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_conv_relu_flatten_gemm import make_conv_relu_flatten_gemm_workload
from stream.inputs.testing.workload.make_resnet_subgraph import make_resnet_subgraph
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload

TWO_CONV = TwoConvWorkloadConfig(1, 8, 32, 32, 16, 32, 3, "bf16", "bf16")

BUILDERS = {
    "2conv": lambda out: make_2_conv_workload(TWO_CONV, out),
    "2conv_mapping": lambda out: make_2_conv_mapping(TWO_CONV, out),
    "small_swiglu": lambda out: make_small_swiglu_workload(output_dir=out),
    "resnet_subgraph": lambda out: make_resnet_subgraph(output_dir=out),
    "conv_relu_flatten_gemm": lambda out: make_conv_relu_flatten_gemm_workload(output_dir=out),
    "aie_gemm": lambda out: make_gemm_workload(64, 64, 64, "bf16", "bf16", out),
    "aie_gemm_mapping": lambda out: make_gemm_mapping(64, 64, 64, 64, 64, 64, output_dir=out),
    "aie_swiglu": lambda out: make_swiglu_workload(256, 512, 2048, "bf16", "bf16", output_dir=out),
    "aie_swiglu_mapping": lambda out: make_swiglu_mapping(256, 512, 2048, True, 32, 32, 64, output_dir=out),
}


@pytest.mark.parametrize("build", BUILDERS.values(), ids=BUILDERS.keys())
def test_a_builder_writes_into_the_directory_given(tmp_path: Path, build) -> None:
    path = Path(build(str(tmp_path)))
    assert path.parent == tmp_path and path.is_file()
