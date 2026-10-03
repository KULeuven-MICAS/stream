"""Workloads lower to an NPU2 design end to end, through the code generation backend that claims the array."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.api import SolveOptions, generate_code  # noqa: E402
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping  # noqa: E402
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping  # noqa: E402
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload  # noqa: E402
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload  # noqa: E402

ACCELERATOR = "stream/inputs/aie/hardware/whole_array_strix.yaml"
SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM = 256, 512, 2048


@pytest.mark.slow
@pytest.mark.parametrize("last_gemm_down", [True, False], ids=["with_down", "ending_in_mul"])
def test_swiglu_generates_an_npu2_design(last_gemm_down: bool, tmp_path):
    workload = make_swiglu_workload(SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, "bf16", "bf16", last_gemm_down=last_gemm_down)
    mapping = make_swiglu_mapping(SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, last_gemm_down, 32, 32, 64)
    options = SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2", "trace_size": 1 << 20})
    module = str(generate_code(ACCELERATOR, workload, str(tmp_path), mapping, options).context.get("module"))
    assert "aie.device(npu2)" in module
    assert len(re.findall(r"aiex?\.", module)) > 100


@pytest.mark.slow
def test_a_single_tile_gemm_generates_an_npu2_design(tmp_path):
    """One kernel tile on one core: no split to unroll and no loop to iterate, so it runs once."""
    workload = make_gemm_workload(64, 64, 64, "bf16", "bf16")
    mapping = make_gemm_mapping(64, 64, 64, 64, 64, 64, nb_rows_to_use=1, nb_cols_to_use=1)
    options = SolveOptions(nb_cols_to_use=1, stage_options={"npu": "npu2"})
    module = str(generate_code(ACCELERATOR, workload, str(tmp_path), mapping, options).context.get("module"))
    assert module.count("func.call @matmul_bf16_bf16_64_64_64") == 1
