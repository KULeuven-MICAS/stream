import dataclasses
import logging
import os
from typing import Any

from zigzag.mapping.temporal_mapping import TemporalMappingType

from stream.cost_model.core_cost import CoreCostEntry
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.mapping.work_share import interchangeable, split_steps
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_backends import (
    COST_BACKEND,
    CoreCostBackend,
    cache_identity,
    discover_backends,
    select_backend,
)
from stream.stages.estimation.zigzag_cost_estimator import SPATIAL_MAPPINGS_GENERATED
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)


class CoreCostEstimationStage(Stage):
    """Computes and caches core cost entries for each valid node-core allocation, through the backend the
    ``core_cost_backend`` stage option names on the cores it claims, else the priority rule's, with each backend's
    ``core_cost_options``; a cached entry is reused only if the same backend priced it with the same cache identity."""

    reads = ("workload", "accelerator", "mapping", "loma_lpf_limit", "output_path", "temporal_mapping_type")
    optional_reads = (
        "nb_spatial_mappings_generated",
        "fusion_splits",
        "loma_show_progress_bar",
        "core_cost_backend",
        "core_cost_options",
    )
    writes = ("cost_lut",)

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        """
        Initialize the stage by:
        - extracting all the unique nodes that will have to be evaluated
        - initializing the valid node-core allocations (which are used later by the InterCoreMappingStage)
        """
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.accelerator: Accelerator = self.ctx.get("accelerator")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.loma_lpf_limit = self.ctx.get("loma_lpf_limit")
        self.output_path = self.ctx.get("output_path")
        self.nb_spatial_mappings_generated: int = self.ctx.get(
            "nb_spatial_mappings_generated", SPATIAL_MAPPINGS_GENERATED
        )
        self.temporal_mapping_type: TemporalMappingType = self.ctx.get("temporal_mapping_type")
        self.fusion_splits: dict = self.ctx.get("fusion_splits", {}) or {}
        self.loma_show_progress_bar: bool = self.ctx.get("loma_show_progress_bar", False)
        self.backends: list[CoreCostBackend] = discover_backends()
        self.core_cost_backend: str | None = self.ctx.get("core_cost_backend")
        self.core_cost_options: dict[str, dict[str, Any]] = self.ctx.get("core_cost_options") or {}
        if unknown := sorted(set(self.core_cost_options) - {backend.name for backend in self.backends}):
            raise ValueError(f"core_cost_options names unknown core-cost backends {unknown}")
        self.identities = {
            backend.name: cache_identity(backend, self.core_cost_options.get(backend.name, {}))
            for backend in self.backends
        }
        self.cost_lut_path: str = os.path.join(self.output_path, "core_cost_lut.pickle")

        self.valid_allocations: dict[ComputationNode, list[Core]] = {}
        for node in self.workload.get_computation_nodes():
            node_mapping = self.mapping.get(node)
            if node_mapping is None:
                raise ValueError(f"No mapping found for node {node.name}")
            assert len(node_mapping.resource_allocation) == 1, (
                "TODO: Support multiple resource allocation entries per node"
            )
            cores = node_mapping.resource_allocation[0]  # TODO: support multiple resource allocation entries
            self.valid_allocations[node] = cores
        self.check_named_backend_claims()
        self.cost_lut: CoreCostLUT = CoreCostLUT(cache_path=self.cost_lut_path, load=True)

    def check_named_backend_claims(self) -> None:
        """Refuse a run whose ``core_cost_backend`` claims none of its cores, as when it is not installed."""
        if self.core_cost_backend is None:
            return
        cores = {core.id: core for allocation in self.valid_allocations.values() for core in allocation}.values()
        if not any(self.backend(core).name == self.core_cost_backend for core in cores):
            raise ValueError(f"core_cost_backend {self.core_cost_backend!r} claims none of the cores this run costs")

    def backend(self, core: Core) -> CoreCostBackend:
        return select_backend(core, self.backends, self.core_cost_backend)

    def key(self, backend: CoreCostBackend) -> tuple[str, Any]:
        return backend.name, self.identities[backend.name]

    def run(self):
        logger.info("Start CoreCostEstimationStage.")
        self.update_cost_lut()
        logger.info("Finished CoreCostEstimationStage.")

        self.ctx.set(cost_lut=self.cost_lut)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def update_cost_lut(self):
        seen_new = False
        for node in self.workload.get_computation_nodes():
            cores = self.valid_allocations[node]
            for core in cores:
                backend = self.backend(core)
                cached = self.cost_lut.has_cost(node, core)
                if cached and self.cost_lut.get_cost(node, core).metadata.get(COST_BACKEND) == self.key(backend):
                    continue
                equal_node = self.cost_lut.get_equal_node(node)
                equal_core = self.cost_lut.get_equal_core(equal_node, core) if equal_node else None
                if equal_core is not None and not interchangeable(
                    self.workload,
                    self.mapping,
                    node,
                    equal_core,
                    core,
                    split_steps(self.workload, self.mapping, node, self.fusion_splits),
                ):
                    equal_core = None
                equal_mapping = self.check_equal_mapping(node, equal_node) if equal_node else None
                equal_cost = self.cost_lut.get_cost(equal_node, equal_core) if equal_node and equal_core else None
                if equal_mapping and equal_cost and equal_cost.metadata.get(COST_BACKEND) == self.key(backend):
                    # Entries are not changed once added, so the copy shares the estimate and owns its metadata
                    cost = dataclasses.replace(equal_cost, metadata=dict(equal_cost.metadata))
                    allow_overwrite = cached or node.name == equal_node.name  # e.g. previous run with same mapping
                    self.cost_lut.add_cost(node, core, cost, allow_overwrite=allow_overwrite)
                    continue
                cost_entry = self.recorded(backend.make(self).estimate(node, core), backend)
                self.cost_lut.add_cost(node, core, cost_entry, allow_overwrite=cached)
                seen_new = True
            self.remove_old_entries(node)
        if seen_new:
            self.cost_lut.save()

    def check_equal_mapping(self, node1: ComputationNode, node2: ComputationNode) -> bool:
        if node2 is None:
            return False
        try:
            eq_node1 = self.mapping.get_equal_computation_node(node1)
            eq_node2 = self.mapping.get_equal_computation_node(node2)
            mapping1 = self.mapping.get(eq_node1) if eq_node1 else None
            mapping2 = self.mapping.get(eq_node2) if eq_node2 else None
        except KeyError:
            return False
        return mapping1 == mapping2

    def remove_old_entries(self, node: ComputationNode):
        # Remove all entries in lut with same name as node but that are not node
        same_name_nodes = [n for n in self.cost_lut.get_nodes() if n.name == node.name and n is not node]
        for n in same_name_nodes:
            self.cost_lut.remove_node(n)

    def recorded(self, entry: CoreCostEntry, backend: CoreCostBackend) -> CoreCostEntry:
        return dataclasses.replace(entry, metadata={**entry.metadata, COST_BACKEND: self.key(backend)})
