"""Tests for get_ir() methods on Mapping and SteadyStateScheduler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from stream.datatypes import LayerDim
from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping


# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------


def make_dim(prefix: str) -> LayerDim:
    """Create a LayerDim with given prefix (used as logical dimension name)."""
    return LayerDim(position=0, prefix=prefix)


def make_core(core_id: int, name: str = "") -> "Core":
    """Create a minimal real Core with id and name."""
    from stream.hardware.architecture.core import Core

    return Core(
        core_id=core_id,
        name=name or f"core_{core_id}",
        core_type="compute",
    )


def make_multicast_path(sources: list[int], targets: list[int], hops: int) -> "MulticastPathPlan":
    """Create a minimal mock MulticastPathPlan with real Core objects."""
    from stream.cost_model.communication_manager import MulticastPathPlan
    from unittest.mock import MagicMock

    plan = MagicMock(spec=MulticastPathPlan)
    plan.sources = tuple(make_core(c) for c in sources)
    plan.targets = tuple(make_core(c) for c in targets)
    plan.total_hops_objective = hops
    # Make isinstance(plan, MulticastPathPlan) work
    plan.__class__ = MulticastPathPlan
    return plan


def make_node(name: str) -> MagicMock:
    """Create a minimal mock Node with a name attribute."""
    node = MagicMock()
    node.name = name
    return node


# ---------------------------------------------------------------------------
# Mapping.get_ir() tests
# ---------------------------------------------------------------------------


class TestMappingGetIr:
    def test_empty_mapping_returns_expected_structure(self):
        """Test 1: Empty Mapping returns dict with empty nodes, fused_groups, runtime_args."""
        mapping = Mapping()
        result = mapping.get_ir()

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "fused_groups" in result
        assert "runtime_args" in result
        assert result["nodes"] == {}
        assert result["fused_groups"] == []
        assert result["runtime_args"] == {}

    def test_mapping_with_core_resource_allocation(self):
        """Test 2: Mapping with ComputationNode mapped to cores returns correct node structure."""
        node = make_node("matmul_0")
        core0 = make_core(0)
        core1 = make_core(1)
        k_dim = make_dim("K")

        node_mapping = NodeMapping(
            resource_allocation=((core0, core1),),
            inter_core_tiling=(((k_dim, 2),),),
            memory_allocation=((core0,),),
        )
        mapping = Mapping(initial={node: node_mapping})
        result = mapping.get_ir()

        assert "matmul_0" in result["nodes"]
        node_ir = result["nodes"]["matmul_0"]
        assert "resource_allocation" in node_ir
        assert "inter_core_tiling" in node_ir
        assert "memory_allocation" in node_ir

        # resource_allocation: list of lists of dicts
        alloc_slot = node_ir["resource_allocation"][0]
        assert alloc_slot[0] == {"type": "core", "id": 0}
        assert alloc_slot[1] == {"type": "core", "id": 1}

        # memory_allocation: list of lists of core IDs
        mem_slot = node_ir["memory_allocation"][0]
        assert mem_slot == [0]

    def test_mapping_with_fused_groups(self):
        """Test 3: Mapping with FusedGroups returns correct fused_groups list."""
        k_dim = make_dim("K")
        n_dim = make_dim("N")
        fused_group = FusedGroup(
            name="group_0",
            layers=("matmul_0", "matmul_1"),
            intra_core_tiling=((k_dim, 4), (n_dim, 2)),
        )
        mapping = Mapping(fused_groups=[fused_group])
        result = mapping.get_ir()

        assert len(result["fused_groups"]) == 1
        fg_ir = result["fused_groups"][0]
        assert fg_ir["name"] == "group_0"
        assert fg_ir["layers"] == ["matmul_0", "matmul_1"]
        assert [str(k_dim), 4] in fg_ir["intra_core_tiling"]
        assert [str(n_dim), 2] in fg_ir["intra_core_tiling"]

    def test_mapping_get_ir_json_round_trip(self):
        """Test 4: json.dumps(mapping.get_ir()) succeeds without TypeError."""
        node = make_node("layer_0")
        core0 = make_core(0)
        core1 = make_core(1)
        m_dim = make_dim("M")
        fused_group = FusedGroup(
            name="group_0",
            layers=("layer_0",),
            intra_core_tiling=((m_dim, 2),),
        )
        node_mapping = NodeMapping(
            resource_allocation=((core0, core1),),
            inter_core_tiling=(((m_dim, 2),),),
            memory_allocation=((core0,),),
        )
        mapping = Mapping(
            initial={node: node_mapping},
            fused_groups=[fused_group],
            runtime_args={"buffer_depth": "4"},
        )
        result = mapping.get_ir()
        # This must not raise
        serialized = json.dumps(result)
        # And must round-trip
        deserialized = json.loads(serialized)
        assert deserialized["nodes"]["layer_0"]["resource_allocation"][0][0] == {"type": "core", "id": 0}

    def test_mapping_with_multicast_path_plan(self):
        """Test 5: Mapping with TransferNode mapped to MulticastPathPlans serializes correctly."""
        from stream.hardware.architecture.core import Core
        from stream.cost_model.communication_manager import MulticastPathPlan

        node = make_node("Transfer(tensor_0)")
        path_plan = make_multicast_path(sources=[0, 1], targets=[2, 3], hops=5)

        node_mapping = NodeMapping(
            resource_allocation=((path_plan,),),
            inter_core_tiling=((),),
            memory_allocation=((),),
        )
        mapping = Mapping(initial={node: node_mapping})
        result = mapping.get_ir()

        node_ir = result["nodes"]["Transfer(tensor_0)"]
        resource_slot = node_ir["resource_allocation"][0]
        path_ir = resource_slot[0]
        assert path_ir["type"] == "path"
        assert path_ir["sources"] == [0, 1]
        assert path_ir["targets"] == [2, 3]
        assert path_ir["hops"] == 5

    def test_mapping_runtime_args_preserved(self):
        """Mapping with runtime_args returns them in get_ir()."""
        mapping = Mapping(runtime_args={"key1": "val1", "key2": "val2"})
        result = mapping.get_ir()
        assert result["runtime_args"] == {"key1": "val1", "key2": "val2"}


# ---------------------------------------------------------------------------
# SteadyStateScheduler.get_ir() tests
# ---------------------------------------------------------------------------


class TestSteadyStateSchedulerGetIr:
    def _make_scheduler(self, **overrides):
        """Create a SteadyStateScheduler with minimal mock dependencies."""
        from stream.cost_model.steady_state_scheduler import SteadyStateScheduler

        workload = MagicMock()
        accelerator = MagicMock()
        mapping = Mapping()
        fusion_splits = {}
        cost_lut = MagicMock()

        # Patch os.makedirs to avoid filesystem side effects
        scheduler = SteadyStateScheduler(
            workload=workload,
            accelerator=accelerator,
            mapping=mapping,
            fusion_splits=fusion_splits,
            cost_lut=cost_lut,
            output_path="",  # empty = no makedirs
            **overrides,
        )
        return scheduler

    def test_pre_solve_sentinel_values(self):
        """Test 1: Pre-solve scheduler returns latency sentinel values of -1."""
        scheduler = self._make_scheduler()
        result = scheduler.get_ir()

        assert isinstance(result, dict)
        assert "latency" in result
        assert result["latency"]["total"] == -1
        assert result["latency"]["per_iteration"] == -1
        assert result["latency"]["overlap_between_iterations"] == -1

    def test_post_solve_latency_values(self):
        """Test 2: After setting latency values, get_ir() returns those exact values."""
        scheduler = self._make_scheduler()
        scheduler.latency_total = 1000
        scheduler.latency_per_iteration = 250
        scheduler.overlap_between_iterations = 50

        result = scheduler.get_ir()
        assert result["latency"]["total"] == 1000
        assert result["latency"]["per_iteration"] == 250
        assert result["latency"]["overlap_between_iterations"] == 50

    def test_backend_and_constraint_selection_in_ir(self):
        """Test 3: get_ir() includes backend (str) and constraint_selection (dict or None)."""
        from stream.opt.solver import ConstraintSelection

        cs = ConstraintSelection(memory_capacity=True, object_fifo_depth=False, buffer_descriptors=True, dma_channels=False)
        scheduler = self._make_scheduler(backend="ORTOOLS_HIGHS", constraint_selection=cs)
        result = scheduler.get_ir()

        assert result["backend"] == "ORTOOLS_HIGHS"
        assert result["constraint_selection"] is not None
        assert result["constraint_selection"]["memory_capacity"] is True
        assert result["constraint_selection"]["object_fifo_depth"] is False
        assert result["constraint_selection"]["buffer_descriptors"] is True
        assert result["constraint_selection"]["dma_channels"] is False

    def test_constraint_selection_none(self):
        """Test 3b: When constraint_selection is None, get_ir() returns null for that field."""
        scheduler = self._make_scheduler(constraint_selection=None)
        result = scheduler.get_ir()
        assert result["constraint_selection"] is None

    def test_mapping_included_in_ir(self):
        """Test 4: get_ir() includes 'mapping' key containing Mapping.get_ir() output."""
        node = make_node("op_0")
        core0 = make_core(0)
        node_mapping = NodeMapping(
            resource_allocation=((core0,),),
            inter_core_tiling=((),),
            memory_allocation=((),),
        )
        mapping = Mapping(initial={node: node_mapping})

        scheduler = self._make_scheduler()
        scheduler.mapping = mapping

        result = scheduler.get_ir()
        assert "mapping" in result
        assert "op_0" in result["mapping"]["nodes"]

    def test_scheduler_get_ir_json_round_trip(self):
        """Test 5: json.dumps(scheduler.get_ir()) succeeds without TypeError."""
        from stream.opt.solver import ConstraintSelection

        cs = ConstraintSelection()
        scheduler = self._make_scheduler(backend="ORTOOLS_GSCIP", constraint_selection=cs)
        scheduler.latency_total = 500
        scheduler.latency_per_iteration = 125
        scheduler.overlap_between_iterations = 25
        scheduler.fusion_splits = {"M": 4, "K": 2}

        result = scheduler.get_ir()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)

        assert deserialized["latency"]["total"] == 500
        assert deserialized["backend"] == "ORTOOLS_GSCIP"

    def test_scheduler_ir_has_required_keys(self):
        """The get_ir() dict contains all required top-level keys."""
        scheduler = self._make_scheduler()
        result = scheduler.get_ir()

        required_keys = {"latency", "backend", "constraint_selection", "fusion_splits", "mapping"}
        assert required_keys.issubset(result.keys())

    def test_fusion_splits_serialized_as_string_keys(self):
        """fusion_splits LayerDim keys are serialized as strings."""
        scheduler = self._make_scheduler()
        scheduler.fusion_splits = {"M": 4, "K": 2}

        result = scheduler.get_ir()
        assert result["fusion_splits"] == {"M": 4, "K": 2}
        # All keys must be strings (JSON-serializable)
        for k in result["fusion_splits"].keys():
            assert isinstance(k, str)
