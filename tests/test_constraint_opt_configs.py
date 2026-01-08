import pytest
from zigzag.datatypes import LayerOperand, MemoryOperand

from stream.opt.allocation.constraint_optimization.allocation import get_optimal_allocations
from stream.opt.allocation.constraint_optimization.config import (
    ComputeMilpConfig,
    ConstraintOptStageConfig,
)
from stream.opt.allocation.constraint_optimization.context import build_constraint_context


class DummyCore:
    def __init__(self, core_id: int, type_name: str):
        self.id = core_id
        self.type = type_name

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"DummyCore({self.id}, {self.type})"


class DummyMemInstance:
    def __init__(self, size: int = 1024):
        self.size = size


class DummyAccelerator:
    def __init__(self, cores, offchip_core_id=None, mem_size: int = 1024):
        self._cores = cores
        self.offchip_core_id = offchip_core_id
        self._mem = DummyMemInstance(mem_size)

    @property
    def core_list(self):
        return list(self._cores)

    def get_core(self, core_id: int):
        return next(c for c in self._cores if c.id == core_id)

    def get_top_instance_of_core(self, core, mem_op: MemoryOperand):
        return self._mem


class DummyTemporalMapping:
    def __init__(self):
        self.mapping_dic_stationary = {LayerOperand("O"): []}


class DummyCostEntry:
    def __init__(self, latency: int, energy: float):
        self.temporal_mapping = DummyTemporalMapping()
        self.latency_total = latency
        self.energy_total = energy


class DummyCostLUT:
    def __init__(self, latency: int = 1, energy: float = 1.0):
        self._cost = DummyCostEntry(latency, energy)

    def get_equal_node(self, node):
        return node

    def get_cost(self, equal_node, core):
        return self._cost


class DummyNode:
    def __init__(self, node_id: int, sub_id: int = 0):
        self.id = node_id
        self.sub_id = sub_id
        self.group = 0
        self.constant_operands = []
        self.memory_operand_links = {}
        self.operand_size_bit = {}
        self.inter_core_tiling = []
        self.chosen_core_allocation = None


class DummyWorkload:
    def __init__(self, nodes):
        self.node_list = list(nodes)

    def edges(self):
        return []

    def subgraph(self, nodes):
        return DummyWorkload(nodes)


def test_compute_milp_rejects_non_compute_core(monkeypatch):
    cores = [DummyCore(0, "compute"), DummyCore(1, "memory")]
    accelerator = DummyAccelerator(cores)
    cfg = ConstraintOptStageConfig()
    ctx = build_constraint_context(accelerator, cfg)
    workload = DummyWorkload([DummyNode(0)])
    cost_lut = DummyCostLUT(latency=1)

    alloc = get_optimal_allocations(
        workload,
        accelerator,
        cost_lut,
        context=ctx,
        compute_config=cfg.compute,
        iterations=1,
    )
    assert {core_id for _, core_id, _ in alloc} == {0}


def test_default_profiles_include_all_compute_roles():
    cores = [DummyCore(0, "compute"), DummyCore(1, "compute")]
    accelerator = DummyAccelerator(cores)
    cfg = ConstraintOptStageConfig()
    ctx = build_constraint_context(accelerator, cfg)
    assert sorted(c.id for c in ctx.compute_cores) == [0, 1]


def test_invalid_compute_config_rejected():
    with pytest.raises(ValueError):
        ComputeMilpConfig(gap=-0.1)
