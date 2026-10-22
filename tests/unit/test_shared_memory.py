import tempfile

import pytest
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.architecture.accelerator import Accelerator
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator

FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
# Fused tiles of the SwiGLU arm of tests/test_hardware_combinations.py: the Gemms run on the array, Silu and Mul on the
# vector core.
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


def solved_model(accelerator: Accelerator) -> AllocationModel:
    solved: list[AllocationModel] = []
    original = AllocationModel.solve

    def solve(self: AllocationModel, *args, **kwargs):
        solved.append(self)
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AllocationModel, "solve", solve)
        solve_swiglu(accelerator)
    return solved[0]


def test_cores_sharing_a_memory_use_one_memory_of_them_both():
    accelerator = fusemax()
    memories = [accelerator.memory_of(accelerator.get_core(i)) for i in (0, 1, 2)]
    assert [m.name for m in memories] == ["sram", "sram", "dram"]
    assert [c.id for c in memories[0].cores] == [0, 1]
    assert len(accelerator.memories) == 2


def test_cores_sharing_a_memory_fit_their_tiles_in_it_together():
    # Together the two cores hold 184 KB here, bounded by the one 300 KB memory.
    rows = solve_swiglu(fusemax(sram_kb=300)).get("allocation").solution.performance["memory_occupancy"]
    (sram,) = [row for row in rows if row["memory"] == "sram"]
    assert sram["cores"] == [0, 1]
    assert sram["resident_bits"] <= sram["capacity_bits"] == 300 * 8192


def test_a_shared_memory_too_small_for_both_cores_is_infeasible():
    with pytest.raises(InfeasibleAllocationError):
        # Apart, each core fits its tiles in 127 KB; together they need 128 KB, the projections sharing one copy of
        # their input.
        solve_swiglu(fusemax(sram_kb=127))


def test_a_handover_between_cores_sharing_a_memory_stays_in_it():
    alloc = solved_model(fusemax())
    handovers = [
        tr
        for tr in alloc.space.transfer_nodes
        if {c.id for choice in alloc.space.path_choices[tr] for c in (*choice.sources, *choice.targets)} == {0, 1}
    ]
    assert handovers
    for tr in handovers:
        for choice in alloc.space.path_choices[tr]:
            assert alloc.space.transfer_latency_for_path(tr, choice) == 0
            assert not alloc.space.links_in_choice[(tr, choice)]
    # Each copy is its source's buffer, and these hold nothing beyond their sources.
    copies = {t.name for tr in handovers for t in tr.outputs}
    beyond = dict.fromkeys(copies, 0)
    for indicator, bits, name in alloc.context.ledger.memory[0]:
        if name in copies and float(indicator.X) > 0.5:
            beyond[name] += bits
    assert all(bits <= 0 for bits in beyond.values())
