import logging
from collections import defaultdict

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import IterationVariable, SteadyStateIterationSpace
from stream.workload.workload import Node, Workload

logger = logging.getLogger(__name__)


class TilingGenerationStage(Stage):
    """
    This stage:
    - Determines the best dimension to fuse the layers on.
    - Substitutes the loop ranges with the smaller tiled ranges.
    - Generates the steady state iteration space for all tensors and computation nodes.
    TODO: Add support for multiple layer stacks. Curently it assumes all layers are fused together.
    """

    REQUIRED_FIELDS = ("accelerator", "workload")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator: Accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.fuse_dimensions: list[str] = []
        self.tiled_sizes: dict[int, int] = {}
        self.steady_state_iteration_spaces: dict[Node, SteadyStateIterationSpace] = {}

    def run(self):
        self.fuse_dimensions = self.determine_fusion_dimensions()
        self.tiled_sizes = self.substitute_loop_sizes_with_tiled_sizes()
        self.steady_state_iteration_spaces = self.generate_steady_state_iteration_spaces()

        self.ctx.set(accelerator=self.accelerator, workload=self.workload, layer_stacks=self.layer_stacks)
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
        unique_dims, _ = self.workload.unique_dimensions()

        ssis_dict = {}

        # Computation Nodes
        for node in self.workload.get_computation_nodes():
            iteration_variables = []
            for dim in unique_dims:
                # Relevancy
                if dim in self.workload.get_dims(node):
                    relevant = True
                else:
                    relevant = False
                # Size
                if dim in self.fuse_dimensions:
                    size = self.workload.get_dimension_size(dim)
                else:
                    size = 1

                iteration_variables.append(IterationVariable(dim, size, relevant))
            ssis = SteadyStateIterationSpace(iteration_variables)
            # nb_macs = prod((self.tiled_sizes[dim] for dim in unique_dims if dim in self.workload.get_dims(node)))
            print(node.name, ssis)
            ssis_dict[node] = ssis

        # Tensors
        for node in self.workload.get_computation_nodes():
            # Inputs
            for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
                pass

            # Output
            output_tensor = node.output
            pass

        return ssis_dict
