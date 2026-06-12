import logging
import os
from math import prod

from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.utils import (
    determine_fusion_splits,
)
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

    REQUIRED_FIELDS = ("workload", "mapping", "output_path")

    # Operator types whose iteration space counts as multiply-accumulate work (for total_mac_ops).
    _MAC_OP_TYPES = ("conv", "gemm", "matmul", "linear")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.output_path: str = self.ctx.get("output_path")
        self.fusion_splits: dict[LayerDim, int] = {}
        self.tiled_sizes: dict[int, int] = {}
        self.steady_state_iteration_spaces: dict[ComputationNode, SteadyStateIterationSpace] = {}
        self.unique_dims, self.dim_expressions = self.workload.unique_dimensions()
        pass

    def run(self):
        self.fusion_splits = determine_fusion_splits(self.workload, self.mapping)
        self.tiled_sizes = self.substitute_loop_sizes_with_tiled_sizes()
        self.tiled_workload = self.workload.with_modified_dimension_sizes(self.tiled_sizes)
        self.tiled_mapping = self.mapping.with_updated_workload(self.tiled_workload, self.workload)

        self.tiled_workload.visualize(os.path.join(self.output_path, "tiled_workload.png"))
        self.ctx.set(
            workload=self.tiled_workload,
            mapping=self.tiled_mapping,
            fusion_splits=self.fusion_splits,
            # Total MAC work of the UNTILED group, for the downstream end-to-end utilization stat.
            total_mac_ops=self._total_mac_ops(),
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _total_mac_ops(self) -> int:
        """Total multiply-accumulate ops in this (untiled) fusion group.

        Product of the full loop-dimension sizes over matmul/conv nodes -- a hardware-independent
        workload property. Consumed by the scheduler to report end-to-end MAC utilization
        (useful MACs / (peak MACs/cycle x total latency)). Must be read from ``self.workload``
        BEFORE tiling shrinks the dimension sizes.
        """
        total = 0
        for node in self.workload.get_computation_nodes():
            if any(k in str(node.type).lower() for k in self._MAC_OP_TYPES):
                total += prod(self.workload.get_dimension_size(d) for d in self.workload.get_dims(node))
        return total

    def substitute_loop_sizes_with_tiled_sizes(self):
        """
        The returned dict maps from dimension to its new total tiled size across the spatial unrollings.
        As such, it can differ from the defined fusion split factor in the mapping input as that is per core.
        """
        unique_dims, _ = self.workload.unique_dimensions()
        result = {}
        # Size for the new tiled dimensions
        for dim, split_factor in self.fusion_splits.items():
            wanted_tile_size, rem = divmod(self.workload.get_dimension_size(dim), split_factor)
            assert rem == 0, (
                f"Dimension size {self.workload.get_dimension_size(dim)} not divisible by "
                f"desired tile size {split_factor}"
            )
            result[dim] = wanted_tile_size
        # Size for non-tiled dimensions
        for dim in set(unique_dims) - set(self.fusion_splits.keys()):
            size = self.workload.get_dimension_size(dim)
            result[dim] = size
        return result

    def _get_total_spatial_unrolling_for_dim(
        self,
        dim: LayerDim,
        spatial_unrollings: set[tuple[LayerDim, int]],
    ) -> int:
        total_unrolling = prod(su[1] for su in spatial_unrollings if su[0] == dim)
        return total_unrolling
