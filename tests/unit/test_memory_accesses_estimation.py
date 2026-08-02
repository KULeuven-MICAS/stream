from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.stages.estimation.memory_accesses_estimation import _flatten_cores


def _core() -> Core:
    return Core.__new__(Core)


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
