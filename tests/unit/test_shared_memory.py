import tempfile

import pytest
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.architecture.accelerator import Accelerator
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator

FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
# Fused tiles of the SwiGLU arm of tests/test_hardware_combinations.py, which split every node over both cores.
TILING = [
    {"dim": "Gemm_Left.D1", "tile": 128},
    {"dim": "Gemm_Down.D2", "tile": 128},
    {"dim": "Gemm_Left.D2", "tile": 32},
    {"dim": "Gemm_Left.D0", "tile": 16},
]


def fusemax(sram_kb: int = 2048) -> Accelerator:
    """Fusemax, whose array (core 0) and vector core (core 1) share one SRAM, shrunk to ``sram_kb``."""
    validator = AcceleratorValidator(open_yaml(FUSEMAX), FUSEMAX)
    assert validator.validate()
    data = validator.normalized_data
    for core in (0, 1):
        data["cores"][core]["memories"]["sram"]["size"] = sram_kb * 8192
    return AcceleratorFactory(data).create()


def solve_swiglu(accelerator: Accelerator):
    workload = make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192)
    options = SolveOptions(stage_options={"intra_core_tiling": TILING})
    with tempfile.TemporaryDirectory() as tmpdir:
        return evaluate_mapping(accelerator, workload, tmpdir, options=options).context


def test_cores_sharing_a_memory_use_the_first_cores():
    accelerator = fusemax()
    assert [accelerator.memory_of(accelerator.get_core(i)).id for i in (0, 1, 2)] == [0, 0, 2]
    assert accelerator.nb_shared_mem_groups == 2


def test_cores_sharing_a_memory_fit_their_tiles_in_it_together():
    # Each core alone holds 242 KB here, so a per-core bound of 300 KB would let them hold 484 KB together.
    rows = solve_swiglu(fusemax(sram_kb=300)).get("scheduler").performance_stats["memory_occupancy"]
    (sram,) = [row for row in rows if row["core_id"] in (0, 1)]
    assert sram["core_id"] == 0
    assert sram["resident_bits"] <= sram["capacity_bits"] == 300 * 8192


def test_a_shared_memory_too_small_for_both_cores_is_infeasible():
    with pytest.raises(InfeasibleAllocationError):
        solve_swiglu(fusemax(sram_kb=200))
