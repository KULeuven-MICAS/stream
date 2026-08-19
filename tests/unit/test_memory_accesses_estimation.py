from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.stages.estimation.memory_accesses_estimation import _flatten_cores, _memory_tile_cores


def _core() -> Core:
    return Core.__new__(Core)


def _typed_core(core_id: int, core_type: str) -> Core:
    core = Core.__new__(Core)
    core.id = core_id
    core.core_type = core_type
    return core


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
    candidates = tuple((_typed_core(i, "compute"),) for i in range(6))
    assert len(_flatten_cores(candidates)) == 6
    assert _memory_tile_cores(candidates) == []


def test_memory_tiles_are_counted_once_each():
    mem0, mem1 = _typed_core(0, "aie2.memory"), _typed_core(1, "aie2.memory")
    allocation = ((mem0, _typed_core(7, "aie2.compute")), (mem1,), (mem0,))
    assert [c.id for c in _memory_tile_cores(allocation)] == [0, 1]
