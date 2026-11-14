"""A copy a transfer lands in the memory of its source is read in place: the memory holds only what it adds."""

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.ir.infeasibility import InfeasibleAllocationError


def test_a_copy_beside_its_source_costs_only_what_it_adds(tmp_path):
    """The attention head's exp output is multicast to the sum on the tensor cores and the div beside it on the vector
    core; the div reads it in place, so the vector core holds it once and the head fits."""
    estimate = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        "stream/inputs/testing/workload/attention_head.onnx",
        str(tmp_path),
    )
    assert estimate.cycles > 0


def test_the_capacity_diagnosis_counts_a_copy_beside_its_source_as_the_constraint_does(tmp_path):
    """A bf16 2048 x 14336 x 4096 Gemm whole on tpu_v7_ironwood, two MXUs per VMEM: each VMEM holds the input,
    58720256 B, 1024 weight columns, 29360128 B, and 2 x 512 output columns, 4194304 B: 92274688 B. The MXUs' copies
    next to their sources add nothing."""
    workload = make_gemm_workload(2048, 14336, 4096, "bf16", "bf16", output_dir=str(tmp_path))
    options = SolveOptions(artifacts=False, stage_options={"intra_core_tiling": [{"dim": "Gemm.D0", "tile": 2048}]})
    with pytest.raises(InfeasibleAllocationError) as raised:
        evaluate_mapping(
            "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml", workload, str(tmp_path), options=options
        )
    unmet = raised.value.report.resources[0].unmet
    assert unmet is not None and unmet.demand_value == 92274688
    assert {term.label: term.value for term in unmet.terms} == {
        "A_1": 58720256,
        "B_1": 29360128,
        "Y_1": 4194304,
        "A_2": 0,
        "B_2": 0,
        "Y_2": 0,
    }
