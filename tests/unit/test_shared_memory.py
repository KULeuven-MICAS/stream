import tempfile

import onnx
import pytest
from onnx import TensorProto, helper
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.ports import OUTPUT
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
    options = SolveOptions(stage_options={"intra_core_tiling": TILING})
    with tempfile.TemporaryDirectory() as tmpdir:
        workload = make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192, output_dir=tmpdir)
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
    memories = [accelerator.memory_of(accelerator.get_core(i), OUTPUT) for i in (0, 1)]
    assert [m.name for m in memories] == ["sram", "sram"]
    assert [c.id for c in memories[0].cores] == [0, 1]
    assert accelerator.offchip_memory.name == "dram"
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
        if {m.name for choice in alloc.space.path_choices[tr] for m in (*choice.sources, *choice.targets)} == {"sram"}
    ]
    assert handovers
    for tr in handovers:
        for choice in alloc.space.path_choices[tr]:
            assert alloc.space.transfer_latency_for_path(tr, choice) == 0
            assert not alloc.space.links_in_choice[(tr, choice)]
    # Each copy is its source's buffer, and these hold nothing beyond their sources.
    copies = {t.name for tr in handovers for t in tr.outputs}
    beyond = dict.fromkeys(copies, 0)
    for indicator, bits, name in alloc.context.ledger.memory[alloc.space.accelerator.memories["sram"]]:
        if name in copies and float(indicator.X) > 0.5:
            beyond[name] += bits
    assert all(bits <= 0 for bits in beyond.values())


def test_a_copy_each_core_sharing_a_memory_reads_apart_is_held_once(tmp_path, solved_model):
    """On fusemax a bf16 3 x 3 convolution of x, 16 -> 16 channels on 32 x 32, runs on the array, the residual add of
    x on the vector core, columns cut to 8. Transfer(x) lands in the shared SRAM at one position per core, but each of
    its copies has one reader on one core: the convolution's copy holds two 10-column windows of 16 x 32, 163840 bits,
    the add's two 8-column tiles, 131072 bits, each once."""
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["y"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["y", "x"], ["z"], name="Add"),
    ]
    shapes = {"x": [1, 16, 32, 32], "w": [16, 16, 3, 3], "z": [1, 16, 32, 32]}
    info = {name: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in shapes.items()}
    graph = helper.make_graph(nodes, "g", [info["x"], info["w"]], [info["z"]])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "residual.onnx")
    options = SolveOptions(artifacts=False, stage_options={"intra_core_tiling": [{"dim": "Conv.D1", "tile": 8}]})
    alloc = solved_model(FUSEMAX, str(tmp_path / "residual.onnx"), str(tmp_path), options=options)
    sram = alloc.space.accelerator.memories["sram"]
    (transfer,) = [tr for tr in alloc.space.transfer_nodes if tr.name == "Transfer(x)"]
    assert [alloc.space.tensor_choices[copy] for copy in transfer.outputs] == [((sram,),), ((sram,),)]
    held = dict.fromkeys((copy.name for copy in transfer.outputs), 0)
    for indicator, bits, name in alloc.context.ledger.memory[sram]:
        if name in held and float(indicator.X) > 0.5:
            held[name] += bits
    assert list(held.values()) == [163840, 131072]
