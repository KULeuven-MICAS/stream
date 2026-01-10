import logging
from collections import defaultdict
from math import prod

from stream.datatypes import InterCoreTiling, LayerDim
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import IterationVariable, SteadyStateIterationSpace
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)


class TilingGenerationStage(Stage):
    """
    This stage:
    - Determines the best dimension to fuse the layers on.
    - Substitutes the loop ranges with the smaller tiled ranges.
    - Generates the steady state iteration space for all tensors and computation nodes.
    TODO: Add support for multiple layer stacks. Curently it assumes all layers are fused together.
    """

    REQUIRED_FIELDS = ("workload", "mapping")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.mapping: Mapping = self.ctx.require_value("mapping", self.__class__.__name__)
        self.fuse_dimensions: list[int] = []
        self.tiled_sizes: dict[int, int] = {}
        self.steady_state_iteration_spaces: dict[ComputationNode, SteadyStateIterationSpace] = {}
        self.unique_dims, _ = self.workload.unique_dimensions()

    def run(self):
        self.fuse_dimensions = self.determine_fusion_dimensions()
        self.tiled_sizes = self.substitute_loop_sizes_with_tiled_sizes()
        self.steady_state_iteration_spaces = self.generate_steady_state_iteration_spaces()
        self.tiled_workload = self.workload.with_modified_dimension_sizes(self.tiled_sizes)

        self.ctx.set(ssis=self.steady_state_iteration_spaces, workload=self.tiled_workload)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def determine_fusion_dimensions(self) -> list[int]:
        """
        Determine the best dimension to fuse the layers on.
        Currently, we fuse on the dimension with the smallest total size across all layers.
        """
        dim_occurrence_count = defaultdict(int)
        for node in self.workload.get_computation_nodes():
            for dim in self.workload.get_dims(node):
                dim_occurrence_count[dim] += 1
        max_dim_count = max(dim_occurrence_count.values())
        max_dims = tuple(k for k, v in dim_occurrence_count.items() if v == max_dim_count)
        for max_dim in max_dims:
            assert dim_occurrence_count[max_dim] == len(self.workload.get_computation_nodes()), (
                "Not all layers share the same dimension for fusion."
            )
        return [max_dims[0]]  # only first dimension for now

    def substitute_loop_sizes_with_tiled_sizes(self):
        max_dims = self.fuse_dimensions
        unique_dims, _ = self.workload.unique_dimensions()
        # Size for the new tiled dimensions
        d = {dim: 1 for dim in max_dims}
        for dim in set(unique_dims) - set(max_dims):
            size = self.workload.get_dimension_size(dim)
            d[dim] = size
        return d

    def generate_steady_state_iteration_spaces(self):
        spatial_unrollings, unique_spatial_unrollings = self._collect_spatial_unrollings()
        iteration_variables = self._create_spatial_iteration_variables(spatial_unrollings, unique_spatial_unrollings)
        temporal_unrollings = self._derive_temporal_unrollings(unique_spatial_unrollings)
        self._add_temporal_iteration_variables(iteration_variables, temporal_unrollings)
        ssis_dict = self._create_steady_state_iteration_spaces(iteration_variables)
        return ssis_dict

    def _create_steady_state_iteration_spaces(self, iteration_variables):
        """Create the steady state iteration spaces for each computation node."""
        ssis_dict: dict[ComputationNode, SteadyStateIterationSpace] = {}
        for node in self.workload.get_computation_nodes():
            ssis_dict[node] = SteadyStateIterationSpace(iteration_variables[node])
            print(node.name, ssis_dict[node])
        return ssis_dict

    def _add_temporal_iteration_variables(self, iteration_variables, temporal_unrollings):
        """Iterate through all computation nodes and add the temporal iteration variables."""
        for node in self.workload.get_computation_nodes():
            for temporal_unrolling in temporal_unrollings:
                dim, size = temporal_unrolling
                relevant = dim in self.workload.get_dims(node)
                iteration_variables[node].append(IterationVariable(dim, size, relevant, spatial=False))

    def _derive_temporal_unrollings(self, unique_spatial_unrollings):
        """Iterate through the unique workload dimensions and get temporal unrollings"""
        temporal_unrollings: list[tuple[LayerDim, int]] = []  # list because order matters
        for dim in self.unique_dims:  # iterate in different order here if needed
            size, rem = divmod(
                self.workload.get_dimension_size(dim),
                self._get_total_spatial_unrolling_for_dim(dim, unique_spatial_unrollings),
            )
            assert rem == 0, (
                f"Dimension size {self.workload.get_dimension_size(dim)} not divisible by spatial unrolling {self._get_total_spatial_unrolling_for_dim(dim, unique_spatial_unrollings)}"
            )
            temporal_unrollings.append((dim, size))
        return temporal_unrollings

    def _create_spatial_iteration_variables(self, spatial_unrollings, unique_spatial_unrollings):
        """Iterate through all computation nodes and add the spatial or
        replacement temporal iteration variables if it doesn't have that spatial unrolling."""
        iteration_variables: dict[ComputationNode, list[IterationVariable]] = defaultdict(list)
        for node in self.workload.get_computation_nodes():
            for spatial_unrolling in unique_spatial_unrollings:
                dim, unrolling = spatial_unrolling
                if spatial_unrolling in spatial_unrollings[node]:
                    # Create a spatial iteration variable
                    iteration_variables[node].append(
                        IterationVariable(
                            dimension=dim,
                            size=unrolling,
                            relevant=True,
                            spatial=True,
                        )
                    )
                elif any(dim == su[0] for su in spatial_unrollings[node]):
                    # This node has a different unrolling size for the unique dim
                    # Create a hybrid of both spatial and temporal iteration variables
                    spatial_size = next(su[1] for su in spatial_unrollings[node] if su[0] == dim)
                    iteration_variables[node].append(
                        IterationVariable(
                            dimension=dim,
                            size=spatial_size,
                            relevant=True,
                            spatial=True,
                        )
                    )
                    remaining_size, rem = divmod(unrolling, spatial_size)
                    assert rem == 0, f"Unrolling size {unrolling} not divisible by spatial size {spatial_size}"
                    iteration_variables[node].append(
                        IterationVariable(
                            dimension=dim,
                            size=remaining_size,
                            relevant=True,
                            spatial=False,
                        )
                    )
                else:
                    # Create a replacement temporal variable
                    relevant = dim in self.workload.get_dims(node)
                    iteration_variables[node].append(
                        IterationVariable(
                            dimension=dim,
                            size=unrolling,
                            relevant=relevant,
                            spatial=False,
                        )
                    )

        return iteration_variables

    def _collect_spatial_unrollings(self):
        spatial_unrollings: dict[ComputationNode, InterCoreTiling] = {}
        for node in self.workload.get_computation_nodes():
            node_mapping = self.mapping.get(node)
            assert node_mapping is not None, f"No mapping found for node {node.name}"
            spatial_unrollings[node] = self._convert_inter_core_tiling_dims(node)

        unique_spatial_unrollings: set[tuple[LayerDim, int]] = set()
        for unrollings in spatial_unrollings.values():
            for unrolling in unrollings:
                # Keep the largest unrolling size for each dimension
                dim, size = unrolling
                existing = next((u for u in unique_spatial_unrollings if u[0] == dim), None)
                if existing is None:
                    unique_spatial_unrollings.add((dim, size))
                elif size > existing[1]:
                    unique_spatial_unrollings.discard(existing)
                    unique_spatial_unrollings.add((dim, size))
        return spatial_unrollings, unique_spatial_unrollings

    def _convert_inter_core_tiling_dims(self, node: ComputationNode) -> InterCoreTiling:
        """Convert inter_core_tiling dimensions from LayerDim to unique workload indices."""
        node_mapping = self.mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        unique_node_dims = self.workload.get_dims(node)
        converted_tiling: InterCoreTiling = []
        for dim, factor in node_mapping.inter_core_tiling:
            dim_idx = dim.get_idx()
            unique_dim = unique_node_dims[dim_idx]
            converted_tiling.append((unique_dim, factor))
        return converted_tiling

    def _get_total_spatial_unrolling_for_dim(
        self,
        dim: LayerDim,
        spatial_unrollings: set[tuple[LayerDim, int]],
    ) -> int:
        total_unrolling = prod(su[1] for su in spatial_unrollings if su[0] == dim)
        return total_unrolling
