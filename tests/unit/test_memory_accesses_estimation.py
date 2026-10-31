from types import SimpleNamespace

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.memory_accesses import CoreMemoryAccesses
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import READ, WRITE
from stream.stages.estimation.memory_accesses_estimation import (
    MemoryAccessesEstimationStage,
    _flatten_cores,
    _memory_tile_cores,
)
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.workload.node import TransferType


def test_an_edge_access_moves_what_the_allocation_models_the_off_chip_memory_at():
    """Off-chip accesses use Strix's measured one-way DRAM rate, or whole_array's widest shim port."""
    strix = parse_accelerator("stream/inputs/aie/hardware/whole_array_strix.yaml")
    dram = strix.offchip_memory
    assert strix.offchip_rate(dram, READ) == strix.offchip_rate(dram, WRITE) == 148.3
    plain = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    dram = plain.offchip_memory
    assert plain.offchip_rate(dram, READ) == plain.offchip_rate(dram, WRITE) == 128


def _core() -> Core:
    return Core.__new__(Core)


def _typed_core(core_id: int, core_type: str) -> Core:
    core = Core.__new__(Core)
    core.id = core_id
    core.core_type = core_type
    return core


def _memory(core: Core) -> Memory:
    """The memory of a tile, its only view."""
    memory = Memory.__new__(Memory)
    memory.name, memory.views, memory.offchip = f"m{core.id}", ((core, "memory"),), False
    return memory


def test_flatten_nested_slots():
    a, b, c = _core(), _core(), _core()
    assert _flatten_cores(((a, b), (c,))) == [a, b, c]


def test_flatten_flat_allocation():
    a, b = _core(), _core()
    assert _flatten_cores((a, b)) == [a, b]


def test_flatten_drops_transfer_paths():
    core = _core()
    path = MulticastPathPlan.__new__(MulticastPathPlan)
    assert _flatten_cores(((core, path),)) == [core]
    assert _flatten_cores((path,)) == []


def test_flatten_empty():
    assert _flatten_cores(None) == []
    assert _flatten_cores(()) == []


def test_only_memory_tiles_are_charged_mem_tile_traffic():
    """On hardware with no mem tiles, no core is charged mem-tile traffic (candidates are compute)."""
    candidates = tuple((_memory(_typed_core(i, "compute")),) for i in range(6))
    assert _memory_tile_cores(candidates) == []


def test_a_staging_memory_compute_cores_also_compute_out_of_charges_its_memory_core():
    """A transfer staged in a VMEM viewed by the VMEM core, MXU and VPU charges only the VMEM core."""
    vmem, mxu, vpu = _typed_core(8, "zigzag.memory"), _typed_core(0, "zigzag.compute"), _typed_core(1, "zigzag.compute")
    memory = Memory.__new__(Memory)
    memory.name, memory.views, memory.offchip = "vmem_0", ((vmem, "vmem"), (mxu, "buf"), (vpu, "buf")), False
    assert memory.kind == "compute" and memory.staging
    assert _memory_tile_cores(((memory,),)) == [vmem]


def test_a_transfer_into_the_memory_its_reader_computes_out_of_stages_nothing():
    """A transfer into a VMEM the MXU reads in place charges the VMEM core no staging accesses."""
    vmem, mxu = _typed_core(8, "zigzag.memory"), _typed_core(0, "zigzag.compute")
    memory = Memory.__new__(Memory)
    memory.name, memory.views, memory.offchip = "vmem_0", ((vmem, "vmem"), (mxu, "buf")), False
    stage = MemoryAccessesEstimationStage.__new__(MemoryAccessesEstimationStage)
    stage.core_memory_accesses = CoreMemoryAccesses(None)
    stage.mapping = SimpleNamespace(get=lambda node: SimpleNamespace(memory_allocation=((memory,),)))
    ssis = SimpleNamespace(get_applicable_temporal_sizes=lambda: [1], reuse_factor=lambda: 1)
    stage.get_mem_core_accesses(SimpleNamespace(transfer_type=TransferType.MEM_TO_COMPUTE, inputs=[], outputs=[]), ssis)
    assert str(stage.core_memory_accesses) == str(CoreMemoryAccesses(None))  # nothing counted


def test_memory_tiles_are_counted_once_each():
    mem0, mem1 = _memory(_typed_core(0, "aie2.memory")), _memory(_typed_core(1, "aie2.memory"))
    allocation = ((mem0, _memory(_typed_core(7, "aie2.compute"))), (mem1,), (mem0,))
    assert [c.id for c in _memory_tile_cores(allocation)] == [0, 1]
