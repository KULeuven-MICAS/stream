import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from math import ceil, prod
from typing import Any

from rtree import index
from zigzag.datatypes import Constants, LayerDim, LayerOperand
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save
from zigzag.workload.layer_attributes import LayerDimSizes
from zigzag.workload.layer_node import LayerNodeAttributes

from stream.cost_model.group_allocation import GroupIdManager
from stream.hardware.architecture.accelerator import Accelerator
from stream.node_tensor import NodeTensor
from stream.opt.partitioning.TemporalLoop import TemporalLoop
from stream.opt.partitioning.utils import convert_outer_cn_loops
from stream.stages.stage import Stage, StageCallable
from stream.utils import contains_wildcard, get_inter_core_tiling_size
from stream.workload.computation.computation_node import LOOP_RANGES_T, ComputationNode, GeneratedComputationNode
from stream.workload.dependency_propagation.dummy_node import DummyNode
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload
from stream.workload.tensor import Tensor

logger = logging.getLogger(__name__)

EDGE_T = tuple[ComputationNode, ComputationNode, dict[str, Any]]


def deduce_tensor_reuse_factors(
    original_node: ComputationNode, outer_temporal_loops: list[TemporalLoop]
) -> dict[LayerOperand, list[int]]:
    """This function is used to generate a list of inter-CN data reuse factor for each CN's constant operand, like W,
      based on the outer-CN loops and the r, ir relations.

    Args:
        original_node (ComputationNode): the original layer node before tilling
        outer_temporal_loops (list[TemporalLoop]): the outer CN temporal loops

    Returns:
        data_reuse_factor (dict[list[int]]): a list of data reuse factor (base priority) for constant operands of each
        CN
    """
    constant_operands = original_node.constant_operands + original_node.partially_constant_operands

    # If there is no loop in the r_ir_loop, meaning that there is no outer-CN loop -> layer-by-layer
    if not outer_temporal_loops:
        return {}

    if not constant_operands:
        return {}

    # Transfer the outer_temporal_loops to r_ir_loop.
    #  An example can be r_ir_loop = {'W': [('ir', 3), ('r', 2), ('ir', 3)]}.
    r_ir_LUT = original_node.loop_relevancy_info
    r_ir_loop: dict[LayerOperand, list[tuple[str, int]]] = {}
    for constant_operand in constant_operands:
        r_ir_loop[constant_operand] = []
        for loop in outer_temporal_loops:
            if loop.dimension in r_ir_LUT.get_ir_layer_dims(constant_operand):
                r_ir_loop[constant_operand].append(("ir", loop.size))
            else:
                r_ir_loop[constant_operand].append(("r", loop.size))

    # total_reuse_factor is the upper bound of the reuse factor that current layer CNs can reach
    total_reuse_factors = {
        op: prod([reuse_factor for (loop_type, reuse_factor) in r_ir_loop[op] if loop_type == "ir"])
        for op in r_ir_loop.keys()
    }

    # total number of nodes that will be generated
    nb_nodes = prod([tl.size for tl in outer_temporal_loops])

    # tensor reuse factor will be set to the total reuse factor for each node
    # whenever a cn will be scheduled, the tensor reuse factor will decrease
    tensor_reuse_factors: dict[LayerOperand, list[int]] = {}
    for op, total_reuse_factor in total_reuse_factors.items():
        tensor_reuse_factors[op] = [total_reuse_factor] * nb_nodes

    return tensor_reuse_factors


class TensorDimensionMismatchError(Exception):
    """Facilitates error handling in case incorrect tensor dimensions are passed on"""


class TiledWorkloadGenerationStage(Stage):
    """
    Class that transforms the layer-by-layer workload into tiled workload graph.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        **kwargs: Any,
    ):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.layer_stacks = kwargs.get("layer_stacks", [])

        # Save for each of the workload's nodes the tiles that will be generated
        self.tiles_dict: dict[ComputationNode, list[ComputationNode]] = {}
        # Memoize the numpy tensors for dependency generation
        self.numpy_tensors: dict[tuple[ComputationNode, LayerOperand], NodeTensor] = {}
        self.tiled_workload_path = tiled_workload_path

    def run(self):
        all_unique_tiles: list[ComputationNode] = []
        # For each node get all the tiles and the edges between them
        all_tiles: list[ComputationNode] = []
        all_edges: list[tuple[ComputationNode, ComputationNode, dict[str, int]]] = []
        for node in self.workload.topological_sort():
            # If other node types shouldn't be included in tiled workload graph, add here
            if not isinstance(node, ComputationNode):
                continue
            outer_temporal_loops = self.get_outer_tmap_loop_dimensions(node)
            tiles, unique_tiles = self.get_tiles(node, outer_temporal_loops)

            # Only log once for generated nodes
            if not isinstance(node, GeneratedComputationNode) or node.gen_id == 0:
                logger.info(f"{node}: Outer loops {outer_temporal_loops}.")
                logger.info(f"{node}: Generated {len(tiles)} tile(s).")
            self.tiles_dict[node] = tiles
            all_unique_tiles += unique_tiles
            intra_edges = self.get_intra_edges(tiles)
            # Add the tiles and intra edges to the lists
            all_tiles += tiles
            all_edges += intra_edges

        # Load in cached tiles and reuse cached tiled_workload if they match
        cached_workload = self.load_cached_tiled_workload()
        if cached_workload and self.cached_workload_matches(all_tiles, cached_workload):
            tiled_workload = cached_workload
            logger.info("Tiled workload loaded from cache.")
        else:
            # Get all pairs of nodes that we have to extract inter edges for
            all_pairs = self.get_all_node_pairs(self.workload)
            for producer, consumer, is_complex in all_pairs:
                if is_complex:
                    inter_edges = self.get_inter_edges_numpy(producer, consumer)
                else:
                    inter_edges = self.get_inter_edges_rtree(producer, consumer)
                all_edges += inter_edges

            # The graph construction needs to happen after the base priority and nb_real_predecessors are set
            tiled_workload = ComputationNodeWorkload()
            tiled_workload.add_nodes_from(all_tiles)
            tiled_workload.add_edges_from(all_edges)

            # Set the base_priority and number of real predecessors of all nodes
            self.set_base_priority_of_nodes(tiled_workload)
            self.set_nb_real_predecessors(tiled_workload)
            tiled_workload = self.remake_workload(all_tiles, all_edges)

            # Save the tiled workload
            pickle_save(tiled_workload, self.tiled_workload_path)  # type: ignore
            logger.info(f"Saved tiled workload to {self.tiled_workload_path}.")

        logger.info(f"Finer graph: {tiled_workload}.")

        kwargs = self.kwargs.copy()
        kwargs["original_workload"] = pickle_deepcopy(self.workload)
        kwargs["workload"] = tiled_workload
        kwargs["accelerator"] = self.accelerator

        if "scheduling_order" not in kwargs:
            kwargs["scheduling_order"] = self.get_scheduling_order(tiled_workload)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()

        yield None, None

    def cached_workload_matches(self, tiles: list[ComputationNode], cached_tiles: ComputationNodeWorkload) -> bool:
        """Check if the tiles match the cached tiled workload.
        Can't use 'has_same_performance' because nb_real_predecessors is not set yet for tiles."""
        logger.info("Checking if the cached workload is valid for this run.")
        # Create hashes of the cached tiles
        cached_tiles_hashes = [tile.static_hash for tile in cached_tiles.node_list]
        # Check for each tile hash if it is in the cached tile hashes
        for tile in tiles:
            if tile.static_hash not in cached_tiles_hashes:
                return False
        return True

    @staticmethod
    def get_scheduling_order(workload: ComputationNodeWorkload):
        return sorted((n.id, n.sub_id) for n in workload.node_list)

    @staticmethod
    def get_all_node_pairs(g: ONNXWorkload) -> tuple[tuple[ComputationNode, ComputationNode, bool], ...]:
        pairs: list[tuple[ComputationNode, ComputationNode, bool]] = []
        for node in g.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            successors = list(g.successors(node))
            is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]
            while not all(is_computation_node):
                non_computation_node_succ_idx = is_computation_node.index(False)
                non_computation_node_succ = successors[non_computation_node_succ_idx]
                succ2 = list(g.successors(non_computation_node_succ))
                successors.pop(non_computation_node_succ_idx)
                successors += succ2
                is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]

            # Now we have all ComputationNode successors
            for successor in successors:
                assert isinstance(successor, ComputationNode), f"Successor {successor} is not a ComputationNode."
                intermediates = g.shortest_path(node, successor)[1:-1]
                complex_pair = False
                for intermediate in intermediates:
                    if isinstance(intermediate, ComputationNode):
                        raise ValueError(
                            "Intermediate node between two ComputationNodes should not be a ComputationNode."
                        )
                    if not isinstance(intermediate, DummyNode):
                        complex_pair = True
                pairs.append((node, successor, complex_pair))
        return tuple(pairs)

    def get_outer_tmap_loop_dimensions(self, node: ComputationNode) -> list[TemporalLoop]:
        """Get the temporal loops that are outside a CN for this node.

        NOTE the order of this list matters! The order in which sub-tiles are generated should match the scheduling
             order. First generate all tiles within the same intra-core split (by splitting inter-core).
             i.e. tiles with sub-id 0, 1, ..., (nb_inter_tiles - 1) should have the same intra-core split and allocated
             to different cores

        Args:
            node: node for which to return outer-cn loops

        Returns:
            temporal loops outside of cn
        """
        if contains_wildcard(node.inter_core_tiling):
            # inter core tiling is not set by CO yet
            tiling_to_split = node.intra_core_tiling
        else:
            # inter core tiling is ok, also split into these tiles. NOTE: this list is ordered
            tiling_to_split = node.inter_core_tiling + node.intra_core_tiling
        outer_loops = convert_outer_cn_loops(tiling_to_split)  # type: ignore

        # In case no valid intra core tiling is found: add an arbitrary tiling of size 1
        if not outer_loops:
            outer_loops = [TemporalLoop(node.layer_dims[0], 1)]

        return outer_loops

    def get_total_outer_size(self, outer_temporal_loops: list[TemporalLoop], dim: LayerDim):
        """Return the total outer temporal size for the given dim."""
        return prod([loop.size for loop in outer_temporal_loops if loop.dimension == dim])

    def get_non_type_predecessors(self, node: Node, types: list[type]) -> list[Node]:
        """Find all self.workload nodes that are not of any type in types.
        If a node of any type in types is a predecessor, we cascade back through the graph until only non-types type
        preds are found.

        Args:
            node (Node): the node for which we intend to find all preds that are not of a type in types
            types (list): list of different types that we want to exclude from our predecessors
        """
        preds: list[Node] = list(self.workload.predecessors(node))
        while any([type(pred) in types for pred in preds]):
            # Find first node in list that is of any type in types
            skip_node = next(pred for pred in preds if any([isinstance(pred, type) for type in types]))
            # Find its index
            idx = preds.index(skip_node)
            # Find its predecessors
            skip_node_preds = list(self.workload.predecessors(skip_node))
            # Pop the skip_node from the list of preds and append its preds to the list
            preds.pop(idx)
            preds += skip_node_preds
        return preds

    def get_mandatory_divisors(self, node: ComputationNode) -> dict[LayerDim, set[int]]:
        """Get the factors by which the (padded) dimensions of this node must be divisible. The dimensions must be
        divisible by the outer loops of all nodes in the same layer stack.  This ensures dependencies between tiles
        within the stack do not cross the layer stack boundaries.
        """
        divisors: dict[LayerDim, set[int]] = defaultdict(lambda: set())

        if isinstance(node, GeneratedComputationNode):
            # Too hard to manage this for generated nodes for now
            return divisors

        # TODO: Discuss with Robin regarding the mandatory divisors
        return divisors

        # Find nodes in stack
        try:
            curr_stack = next(stack for stack in self.layer_stacks if node.id in stack)
        except StopIteration:
            # No stack found
            return divisors
        if len(curr_stack) == 1:
            return divisors
        other_nodes_in_stack = [
            n
            for n in self.workload.node_list
            if n.id in curr_stack and n.id != node.id and isinstance(n, ComputationNode)
        ]

        for other_node in other_nodes_in_stack:
            outer_sizes = self.get_outer_tmap_loop_dimensions(other_node)
            for layer_dim in other_node.layer_dims:
                total_outer_size = self.get_total_outer_size(outer_sizes, layer_dim)
                divisors[layer_dim].add(total_outer_size)
        return divisors

    def get_tiles(
        self,
        original_node: ComputationNode,
        outer_temporal_loops: list[TemporalLoop],
    ) -> tuple[list[ComputationNode], list[ComputationNode]]:
        mandatory_divisors = self.get_mandatory_divisors(original_node)
        # Pad the layer_dim_sizes to be divisible by the mandatory divisors (coming from the outer_temporal_loops)
        tile_attrs = original_node.extract_node_attr()
        tile_attrs = self._pad_layer_dim_sizes(tile_attrs, outer_temporal_loops, mandatory_divisors)
        # Save these extended sizes for later
        original_node.extended_layer_dim_sizes = deepcopy(tile_attrs.layer_dim_sizes)

        # Take away the outer temporal loops to create tiled CNs for this node
        tile_attrs = self._take_away_outer_temporal_loops(tile_attrs, outer_temporal_loops)

        # Loop dimension + size of the tiles (called span here)
        tile_span = tile_attrs.layer_dim_sizes
        stop_values = [temporal_loop.size for temporal_loop in outer_temporal_loops]
        nb_cns = int(prod(stop_values))

        # Compute the data_reuse_factor (will be used as base_priority later) for the constant operands of all CNs
        tensor_reuse_factors = deduce_tensor_reuse_factors(original_node, outer_temporal_loops)
        mult_factors = self._get_multiplication_factors(outer_temporal_loops, tile_span)

        tiles: list[ComputationNode] = []
        tensors: list[Tensor] = []
        group_id_manager = GroupIdManager(
            layer_dim_sizes=original_node.extended_layer_dim_sizes,
            intra_core_tiling=original_node.intra_core_tiling,
            inter_core_tiling=original_node.inter_core_tiling,  # type: ignore
        )

        for n in range(nb_cns):
            dim_min_max = self._get_dim_min_max(
                original_node=original_node,
                outer_temporal_loops=outer_temporal_loops,
                tile_span=tile_span,
                multiplication_factors=mult_factors,
                stop_values=stop_values,
                n=n,
            )
            group_id = group_id_manager.get_group_id(dim_min_max)
            produces_final_output = self.produces_final_output(original_node, dim_min_max)

            tile = self.create_tile(
                original_node,
                sub_id=n,
                tile_attrs=tile_attrs,
                produces_final_output=produces_final_output,
                group_id=group_id,
            )

            # Override loop_ranges property
            tile.update_loop_ranges(dim_min_max)

            # Initialize the priorities (total inter-CN data reuse factor) for the constant operands of this tile
            for constant_operand in tile.constant_operands + tile.partially_constant_operands:
                tensor = tile.operand_tensors[constant_operand]
                tensor.set_base_priorities(tensor_reuse_factors[constant_operand][n])

            tensors = self._replace_identical_tensors(tile, tensors)
            tile.data_produced_unique = self._get_data_produced_unique(tile)
            self._set_core_allocation_for_tile(tile, group_id, original_node)
            tiles.append(tile)

        # NOTE We take the first node as only unique one as they are all generated equally now.
        unique_tiles = [tiles[0]]

        return tiles, unique_tiles

    def pad_until_divisible(
        self,
        layer_dim: LayerDim,
        n: int,
        outer_temporal_loops: list[TemporalLoop],
        mandatory_divisors: dict[LayerDim, set[int]],
    ) -> int:
        """Return x >= n such that x is divisible by `total_outer_size`, as well as by all `mandatory_divisors`
        (coming from the inter-core tiling of other nodes within the same stack)"""
        total_outer_size = self.get_total_outer_size(outer_temporal_loops, layer_dim)
        all_divisors = list(mandatory_divisors[layer_dim]) + [total_outer_size]

        for divisor in all_divisors:
            if n % divisor != 0:
                n = ceil(n / divisor) * divisor
        return n

    def _pad_layer_dim_sizes(
        self,
        tile_attrs: LayerNodeAttributes,
        outer_temporal_loops: list[TemporalLoop],
        mandatory_divisors: dict[LayerDim, set[int]],
    ):
        """Pad layer_dim_sizes to be divisible by the mandatory divisors."""

        for dim, size in tile_attrs.layer_dim_sizes.items():
            new_size = self.pad_until_divisible(dim, int(size), outer_temporal_loops, mandatory_divisors)
            if size != new_size:
                tile_attrs.layer_dim_sizes[dim] = new_size
                logger.warning(f"Padded layer dimension {dim}: {size} -> {new_size} to be divisible by tiling factors")
        return tile_attrs

    def _take_away_outer_temporal_loops(
        self, tile_attrs: LayerNodeAttributes, outer_temporal_loops: list[TemporalLoop]
    ):
        """Take away the outer_temporal_loops to create tiled CNs for this node"""
        for loop in outer_temporal_loops:
            outer_dim, outer_size = loop.unpack()
            node_dim_size: int = tile_attrs.layer_dim_sizes[outer_dim]
            q, rem = divmod(node_dim_size, outer_size)  # returns x//y, x%y
            assert rem == 0, "Should be guaranteed through mandatory divisors"
            tile_attrs.layer_dim_sizes[outer_dim] = q
        return tile_attrs

    def _get_multiplication_factors(
        self, outer_temporal_loops: list[TemporalLoop], tile_span: LayerDimSizes
    ) -> list[int]:
        """Multiplication factor for each outer-cn loop.
        This is to convert from the relative loop value which goes from 0, 1, ..., stop_value - 1
        to the absolute value of that dimension (if there is another lower loop of the same type or spatial loop)"""
        mult_factors: list[int] = []
        for i, loop in enumerate(outer_temporal_loops):
            loop_dim, stop_value = loop.unpack()
            inner_span = tile_span[loop_dim] if loop_dim in tile_span else 1
            lower_outer_cn_loops = outer_temporal_loops[:i]
            # Returns 1 if empty list
            outer_span = prod(
                [temporal_loop.size for temporal_loop in lower_outer_cn_loops if temporal_loop.dimension == loop_dim]
            )
            mult_factors.append(int(inner_span * outer_span))
        return mult_factors

    def _get_outer_loop_values(
        self, n: int, outer_temporal_loops: list[TemporalLoop], stop_values: list[int]
    ) -> list[int]:
        outer_loop_values: list[int] = []
        for i, outer_loop in enumerate(outer_temporal_loops):
            stop_value = outer_loop.size
            m = prod(stop_values[:i])
            outer_loop_values.append((n // m) % stop_value)
        return outer_loop_values

    def _get_dim_min_max(
        self,
        original_node: ComputationNode,
        n: int,
        outer_temporal_loops: list[TemporalLoop],
        tile_span: LayerDimSizes,
        stop_values: list[int],
        multiplication_factors: list[int],
    ) -> LOOP_RANGES_T:
        outer_loop_values = self._get_outer_loop_values(n, outer_temporal_loops, stop_values)

        dim_min_max: LOOP_RANGES_T = {}
        for loop_dim in original_node.layer_dims:
            # multiply all outer-cn loop values that iterate over this loop_dim by their mult_factor
            dim_min = 0
            for i, outer_loop in enumerate(outer_temporal_loops):
                if outer_loop.dimension == loop_dim:
                    # current loop value of this outer-cn loop
                    loop_val = outer_loop_values[i]
                    # mult factor of this outer-cn loop
                    mult_factor = multiplication_factors[i]
                    dim_min += loop_val * mult_factor
            # max value is exclusive
            dim_max = dim_min + (tile_span[loop_dim] if loop_dim in tile_span else 1)
            dim_min_max[loop_dim] = (dim_min, dim_max)
        return dim_min_max

    def produces_final_output(self, node: ComputationNode, dim_min_max: LOOP_RANGES_T) -> bool:
        """If all the output irrelevant loops are at a max, this is producing a final output, so set a flag"""
        original_node_output_ir_dims = node.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
        return all([dim_min_max[dim][1] >= node.layer_dim_sizes[dim] for dim in original_node_output_ir_dims])

    def _replace_identical_tensors(self, tile: ComputationNode, previous_tensors: list[Tensor]):
        """Replace any of the tensors with identical tensors of previous tiles.
        If no identical tensor is found, add the tensor to the list of previous tensors."""
        for op, tensor in tile.operand_tensors.items():
            replaced = False
            for previous_tensor in previous_tensors:
                if tensor.equality_hash == previous_tensor.equality_hash:
                    tile.operand_tensors[op] = previous_tensor
                    replaced = True
            if not replaced:
                previous_tensors.append(tensor)
        return previous_tensors

    def _get_data_produced_unique(self, tile: ComputationNode):
        """Compute the output data produced by each tile, assuming that all the data produced by different CNs unique"""
        return int(
            tile.operand_size_elem[Constants.OUTPUT_LAYER_OP] * tile.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
        )

    def _set_core_allocation_for_tile(self, tile: ComputationNode, group_id: int, original_node: ComputationNode):
        """If the core allocation is fixed, we need to set the chosen core allocation. It's possible the core allocation
        contains multiple entries. In that case, we select the core allocation based on the group id.
        Only set the core allocation if the number of core allocations is equal to the inter-core tiling size, i.e.
        the user meant to parallelize the original nodes over the given cores. Otherwise, the CO or GA will set the
        allocation later."""
        inter_core_tiling_size = get_inter_core_tiling_size(original_node)
        if len(original_node.possible_core_allocation) == inter_core_tiling_size:
            assert group_id < len(original_node.possible_core_allocation), (
                f"Group id {group_id} too large for core allocation list {original_node.possible_core_allocation}"
            )
            chosen_core_allocation = original_node.possible_core_allocation[group_id]
            tile.set_chosen_core_allocation(chosen_core_allocation)

    def create_tile(
        self,
        original_node: ComputationNode,
        sub_id: int,
        tile_attrs: LayerNodeAttributes,
        produces_final_output: bool,
        group_id: int,
    ):
        if isinstance(original_node, GeneratedComputationNode):
            return GeneratedComputationNode(
                node_id=original_node.id,
                sub_id=sub_id,
                base_id=original_node.base_id,
                gen_id=original_node.gen_id,
                gen_split_layer_dim=original_node.gen_split_layer_dim,
                node_name=original_node.name,
                node_attr=tile_attrs,
                mapping_attr=original_node.extract_inter_core_mapping_attr(),
                op_type=original_node.type,
                produces_final_output=produces_final_output,
                group_id=group_id,
            )
        else:
            return ComputationNode(
                node_id=original_node.id,
                sub_id=sub_id,
                node_name=original_node.name,
                node_attr=tile_attrs,
                mapping_attr=original_node.extract_inter_core_mapping_attr(),
                op_type=original_node.type,
                produces_final_output=produces_final_output,
                group_id=group_id,
                partially_constant_operands=original_node.partially_constant_operands,
            )

    @staticmethod
    def get_intra_edges(nodes: list[ComputationNode]):
        """
        # TODO Why do we need this?
        """
        # Get all the group ids
        group_ids = sorted(set([n.group for n in nodes]))
        intra_edges: list[tuple[ComputationNode, ComputationNode, dict[str, int]]] = []
        for group_id in group_ids:
            group_nodes = [n for n in nodes if n.group == group_id]
            pairs = zip(group_nodes, group_nodes[1:], strict=False)
            for node_1, node_2 in pairs:
                intra_edges.append((node_1, node_2, {"bits": 0}))
        return intra_edges

    def convert_to_inclusive_data_range(self, exclusive_data_range: LOOP_RANGES_T):
        """
        Convert an exclusive data range to an inclusive data range.
        """
        return {key: (min_val, max_val - 1) for key, (min_val, max_val) in exclusive_data_range.items()}

    def get_bounding_box_dimensions(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
        dimensions: list[LayerDim],
        loop_ranges: LOOP_RANGES_T,
        interleaved: bool = True,
    ) -> tuple[int, ...]:
        """
        Extract the relevant dimension ranges for building the rtree with the dimensions in dimensions.
        The order of the operand's dimensions is determined through the dimensions parameter.
        """
        # Add compensation for grouped convolutions:
        # If there is a G dimension in the loop ranges alongside a C or K, it means we have a 5D tensor,
        # where the onnx tensors are always flattened back to 4D (merging the G+C or G+K into one channel dimension)
        dimensions, loop_ranges = self.flatten_grouped_convolution_ranges(producer, consumer, dimensions, loop_ranges)
        bounding_box = [loop_ranges[dim] for dim in dimensions]
        # TODO can bounding box have size 1? Will probably crash if so

        if not interleaved:
            bounding_box_flat = tuple([item for sublist in bounding_box for item in sublist])
            return bounding_box_flat
        else:
            bounding_box_flat = tuple(zip(*bounding_box, strict=False))
            bounding_box_flat = tuple([item for sublist in bounding_box_flat for item in sublist])
            return bounding_box_flat

    def bounding_box_generator(
        self, producer: ComputationNode, consumer: ComputationNode, nodes: list[ComputationNode], operand: LayerOperand
    ):
        """
        Generator function that yields the bounding boxes of an operand for all nodes.
        """
        for i, node in enumerate(nodes):
            inclusive_ranges = self.convert_to_inclusive_data_range(node.loop_ranges)
            dimensions = node.operand_dimensionality_order[operand]
            bounds = self.get_bounding_box_dimensions(producer, consumer, dimensions, inclusive_ranges)

            # TODO this is a whacky fix
            # RTree doesn't accept bound of one dimension
            BOUNDS_LENGTH_FOR_SINGLE_DIMENSION = 2
            if len(bounds) == BOUNDS_LENGTH_FOR_SINGLE_DIMENSION:
                bounds = (0, 0) + bounds

            yield (i, bounds, None)

    def get_nb_input_dimensions(self, node: ComputationNode, operand: LayerOperand):
        """Return the number of input dimensions this node has.
        # We take the first non-constant input operand."""
        dims = node.operand_dimensionality_order[operand]

        if LayerDim("G") in dims and (LayerDim("C") in dims or LayerDim("K") in dims):
            # because later the generator will merge them into a single channel dim
            return len(dims) - 1
        else:
            return len(dims)

    def build_rtree(
        self, producer: ComputationNode, consumer: ComputationNode, nodes: list[ComputationNode], operand: LayerOperand
    ):
        """
        Build an rtree data structure based on each node in 'nodes' for the relevant dimensions of operand.
        """
        props = index.Property()
        # We assume all nodes in 'nodes' have identical dimensions
        props.dimension = max(self.get_nb_input_dimensions(nodes[0], operand), 2)

        rtree = index.Index(self.bounding_box_generator(producer, consumer, nodes, operand), properties=props)
        return rtree

    def flatten_grouped_convolution_ranges(
        self, producer: ComputationNode, consumer: ComputationNode, dims: list[LayerDim], ranges: LOOP_RANGES_T
    ):
        """If both C/K and G are present in dimensions, flatten their loop ranges so the tensor is 4D.

        Args:
            dimensions (list): list of the different tensor dimensions
            loop_ranges: dict of the loop ranges for the current node.
        """
        # TODO these should be constants
        dim_G = LayerDim("G")
        dim_C = LayerDim("C")
        dim_K = LayerDim("K")
        dim_CH = LayerDim("CH")

        dims_copy = deepcopy(dims)
        ranges_copy = deepcopy(ranges)
        assert all([dim in ranges_copy for dim in dims_copy])

        if dim_G in dims_copy and (dim_C in dims_copy or dim_K in dims_copy):
            G_idx = dims_copy.index(dim_G)
            if dim_C in dims_copy:
                is_consumer = True
                C_K_idx = dims_copy.index(dim_C)
            elif dim_K in dims_copy:
                C_K_idx = dims_copy.index(dim_K)
                is_consumer = False
            else:
                return dims_copy, ranges_copy
            # Replace the G + C/K into one dimension we call "CH" (name doesn't really matter)
            (G_min, G_max_incl) = ranges_copy[dim_G]
            (C_K_min, C_K_max_incl) = ranges_copy[dims_copy[C_K_idx]]
            CH_min = G_min + C_K_min
            original_node = consumer if is_consumer else producer
            CH_max_incl = G_max_incl * original_node.layer_dim_sizes[dims_copy[C_K_idx]] + C_K_max_incl
            ranges_copy[LayerDim("CH")] = (CH_min, CH_max_incl)

            # Remove the G + C/K from the original dimensions list and add CH in its place
            min_idx = min(G_idx, C_K_idx)

            dims_copy.remove(dim_G)
            second_dim = dim_C if is_consumer else dim_K
            dims_copy.remove(second_dim)
            # insert it in place of G or C/K, whichever came first
            dims_copy.insert(min_idx, dim_CH)

        assert all([dim in ranges_copy for dim in dims_copy])
        return dims_copy, ranges_copy

    def get_inter_edges_rtree(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
    ):
        """Function that finds the edges between producer and consumer tiles.

        Args:
            producer: the producer node
            consumer: the consumer node
        """
        producer_tiles = self.tiles_dict[producer]
        consumer_tiles = self.tiles_dict[consumer]

        # Check all the different input operands of the consumer node that stem from the producer node
        # The direct predecessor of an input operand might be a DummyNode so we need to propagate back
        dependent_input_operands: list[LayerOperand] = []
        for operand, parent_node_id in consumer.input_operand_source.items():
            parent_node = self.workload.get_node_with_id(parent_node_id)
            if parent_node == producer:
                dependent_input_operands.append(operand)
            elif not isinstance(parent_node, ComputationNode):
                # Propagate to the first parent CN
                non_dummy_parents = self.get_non_type_predecessors(parent_node, [DummyNode])
                if producer in non_dummy_parents:
                    dependent_input_operands.append(operand)

        # edges will hold the cns that are dependent on each other [(prod_cn, cons_cn), ...]
        edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []

        for input_operand in dependent_input_operands:
            # Build the tree of all consumer tiles for this operand
            consumer_tree = self.build_rtree(producer, consumer, consumer_tiles, input_operand)

            # As long as we haven't iterated through all of the output's operand's irrelevant dimensions,
            # we shouldn't add an edge to the consumer layer's nodes, as this would create unnecessary graph complexity
            # Because we have the intra-edges between the nodes, and because the nodes irrelevant loops are
            # incrementing, we can make the graph simpler by just having one edge at the final irrelevant loop iteration
            # producer node. # Get the relevant (including partially relevant) and irrelevant dimensions of the producer
            # node's output
            producer_r_dims_output = producer.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]
            producer_ir_dims_output = producer.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)

            # Iterate through all the producer nodes and get the consumer nodes that require its outputs,
            # taking into account that we only want an edge if the producer's irrelevant loops are at a max
            for producer_tile in producer_tiles:
                # Get the output irrelevant loop ranges and check if they are at least at the max
                ir_dims_not_at_max = [
                    producer_tile.loop_ranges[ir_dim][1] < producer.loop_ranges[ir_dim][1]
                    for ir_dim in producer_ir_dims_output
                ]
                if any(ir_dims_not_at_max):
                    continue  # to the next producer tile

                p_inclusive_ranges = self.convert_to_inclusive_data_range(producer_tile.loop_ranges)
                p_bounding_box = self.get_bounding_box_dimensions(
                    producer, consumer, producer_r_dims_output, p_inclusive_ranges
                )

                # Get the consumer tile ids that intersect with this producer tile
                intersecting_consumer_node_ids = consumer_tree.intersection(p_bounding_box)

                for intersecting_consumer_node_id in intersecting_consumer_node_ids:
                    intersecting_consumer = consumer_tiles[intersecting_consumer_node_id]
                    # Create a new communication node that will reside between the producer and consumer node
                    edges += [
                        (
                            producer_tile,
                            intersecting_consumer,
                            {
                                "operand": input_operand,
                                "bits": producer_tile.data_produced_unique,
                            },
                        )
                    ]

        return edges

    def get_inter_edges_numpy(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
    ):
        all_inter_edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []
        paths_between = self.workload.find_paths_with_intermediate_type(producer, consumer, PropagationNode)
        assert len(paths_between) > 0, (
            "No paths between producer and consumer found without ComputationNode in intermediates."
        )

        for path_between in paths_between:
            timesteps = (time.time(),)
            # First node in the path is a ComputationNode, of which we extract the output operand dependency tensor
            first_node = path_between[0]
            assert isinstance(first_node, ComputationNode), "First node in path should be ComputationNode"
            tensor = self.get_tensor_cn_for_op(first_node, dependent_operand=Constants.OUTPUT_LAYER_OP)
            timesteps += (time.time(),)

            # Propagate through intermediate, non-computation nodes
            relevant_axes = self._initialize_relevant_axes(first_node, tensor)
            for i, node in enumerate(path_between[1:-1], start=1):
                assert isinstance(node, PropagationNode), "Intermediate nodes should not be of type ComputationNode"
                previous_node = path_between[i - 1]
                next_node = path_between[i + 1]
                tensor, relevant_axes = node.propagate(
                    tensor, previous_node=previous_node, next_node=next_node, relevant_axes=relevant_axes
                )

            # Final node: Computation node
            final_node: ComputationNode = path_between[-1]  # type: ignore
            assert isinstance(final_node, ComputationNode), "Last node in path should be ComputationNode"
            dependent_operand = self._get_dependent_operand(node, final_node)
            inter_edges = self.get_inter_edges_hybrid(tensor, final_node, dependent_operand, relevant_axes)

            timesteps += (time.time(),)
            self._print_time_delta_to_logger(timesteps, str(path_between))

            for p, c in inter_edges:
                all_inter_edges.append(
                    (
                        p,
                        c,
                        {
                            "operand": dependent_operand,
                            "bits": p.data_produced_unique,
                        },
                    )
                )
        return all_inter_edges

    def _print_time_delta_to_logger(self, timesteps: tuple[float, float, float], path: str):
        ts_deltas = [timesteps[i] - timesteps[i - 1] for i in range(1, len(timesteps))]
        ts_deltas_str = ", ".join([f"{delta:.3f}" for delta in ts_deltas])
        logger.debug(f"Path {path} time deltas: {ts_deltas_str}")

    def _get_dependent_operand(self, producer: Node, consumer: ComputationNode):
        """Find the operand for which the consumer node connects to its predecessor"""
        return next(
            op for op, dependent_node_id in consumer.input_operand_source.items() if dependent_node_id == producer.id
        )

    def _initialize_relevant_axes(self, node: ComputationNode, tensor: NodeTensor):
        """The relevant axes represent which tensor ranks are relevant for the dependency propagation, i.e. which axes
        of the NodeTensor are actually used.
        Axes of dimensions appearing in the inter- or intra-core tiling are relevant by default.
        """
        dimensions = node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]
        relevant_axes = [False] * len(dimensions)

        tilings = node.intra_core_tiling
        if not contains_wildcard(node.inter_core_tiling):
            tilings += node.inter_core_tiling

        for dim, size in tilings:
            assert isinstance(size, int), f"Size of tiling {dim} should be an integer, got {size}"
            if dim in dimensions and size > 1:
                relevant_axes[dimensions.index(dim)] = True

        return relevant_axes

    def get_tensor_cn_for_op(self, node: ComputationNode, dependent_operand: LayerOperand):
        """Convert the given node into a NodeTensor and update the known tensors of computation nodes"""
        if (node, dependent_operand) in self.numpy_tensors:
            tensor = self.numpy_tensors[(node, dependent_operand)]
        else:
            tiles = self.tiles_dict[node]
            tensor = self.get_node_tensor(node, tiles, dependent_operand)
            # Store result for later use
            self.numpy_tensors[(node, dependent_operand)] = tensor
        return tensor

    def get_inter_edges_hybrid(
        self, tensor: NodeTensor, final_node: ComputationNode, op: LayerOperand, relevant_axes: list[bool]
    ):
        """This method obtains the tile dependencies between producers in tensor and the consumer final_node.
        This is done by iterating through all consumer tiles,
        for each consumer node we create a window and get all the producer nodes that produced this data window.

        Args:
            tensor (NodeTensor): A tensor containing for each position which CNs will produce it
            final_node (ComputationNode): The node for which to get the inter-edges
            operand (LayerOperand): The input operand of final_node for which to get the inter-edges
            relevant_axes (list): A list of boolean values indicating which axes are relevant for the final_node
        """
        inter_edges: set[tuple[ComputationNode, ComputationNode]] = set()
        dims = final_node.operand_dimensionality_order[op]
        assert len(dims) == len(relevant_axes)
        for consumer_tile in self.tiles_dict[final_node]:
            relevant_loop_ranges = [consumer_tile.loop_ranges[dim] for dim in dims]
            # Override loop ranges of irrelevant axes to only include a single slice
            for i, relevant in enumerate(relevant_axes):
                if not relevant:
                    relevant_loop_ranges[i] = (0, 1)
            # Ellipsis adds the entire last axis for the extra dimension in NodeTensor
            slices = tuple(slice(start, stop) for start, stop in relevant_loop_ranges) + (Ellipsis,)
            sliced_tensor = tensor[slices]
            producer_tiles = set(sliced_tensor[sliced_tensor != 0].flat.flat)  # type: ignore

            for producer_tile in producer_tiles:
                inter_edges.add((producer_tile, consumer_tile))
        return inter_edges

    @staticmethod
    def get_inter_edges_tensor_based(producer_output_tensor: NodeTensor, consumer_input_tensor: NodeTensor):
        """This method obtains the edges between a producer and consumer.
        This is done by iterating through all consumer tiles,
        for each consumer node we create a window and get all the producer nodes that produced this data window.

        Args:
            producer_output_tensor (np.ndarray): A tensor containing for each position which CNs will produce it
            consumer_input_tensor (np.ndarray): A tensor containing for each position which CNs will consume it
        """
        if producer_output_tensor.tensor_shape != consumer_input_tensor.tensor_shape:
            raise TensorDimensionMismatchError("Arrays to construct inter-layer edges must be equal shape.")

        inter_edges: set[tuple[ComputationNode, ComputationNode]] = set()
        for producer_array, consumer_array in zip(
            producer_output_tensor.flat, consumer_input_tensor.flat, strict=False
        ):
            for producer in producer_array:
                # The producer/consumer array may contain a lot of 0
                if not producer:
                    continue
                for consumer in consumer_array:
                    if not consumer:
                        continue

                    inter_edges.add((producer, consumer))
        return inter_edges

    def get_node_tensor(
        self,
        node: ComputationNode,
        tiles: list[ComputationNode],
        op: LayerOperand,
    ) -> NodeTensor:
        tensor_dims = node.operand_dimensionality_order[op]
        assert node.pr_layer_dim_sizes is not None, f"Node {node} must have pr_layer_dim_sizes set"
        all_loop_dim_sizes = node.layer_dim_sizes + node.pr_layer_dim_sizes  # union
        tensor_shapes: tuple[int, ...] = tuple(all_loop_dim_sizes[dim] for dim in tensor_dims)

        if op == node.output_operand:
            ir_dims_output = node.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
            tile_list = tiles  # list in regular order
            should_add_to_tensor_list = [
                all(tile.loop_ranges[ir_dim][1] >= node.loop_ranges[ir_dim][1] for ir_dim in ir_dims_output)
                for tile in tile_list
            ]
        else:
            tile_list = list(reversed(tiles))  # list in reversed order
            should_add_to_tensor_list = [True for _ in tile_list]
            # if this layer is the first layer, we assume the inputs are streamed and "free"

        node_tensor = NodeTensor.initialize_empty(tensor_shapes)
        for tile, should_add_to_tensor in zip(tile_list, should_add_to_tensor_list, strict=False):
            if not should_add_to_tensor:
                continue  # Skip if we're not at the max ir loop value for output

            op_dim_ranges = [tile.loop_ranges[loop_dim] for loop_dim in tensor_dims]
            op_dim_ranges_max_stop = tuple(tensor_shapes)

            # Set this window of the tensor to indicate it will be consumed/produced by this tile
            # NOTE assert is not guaranteed: tiles of range extended nodes can exceed the NodeTensor shape
            # assert all(start < max_stop for (start, _), max_stop in zip(op_dim_ranges, op_dim_ranges_max_stop))

            # Slices that exceed the max stop are reduced to a size-1 slice at `max_stop-1`
            bounded_op_dim_ranges = tuple(
                slice(max(0, min(max_stop - 1, start)), min(max_stop, stop))
                for ((start, stop), max_stop) in zip(op_dim_ranges, op_dim_ranges_max_stop, strict=False)
            )
            node_tensor = node_tensor.extend_with_node(bounded_op_dim_ranges, tile)

        # The dimensionality order of this input/output operand might include
        # both a G and C/K dimension because the ComputationNode gets the group as an extra
        # dimension in its input/output operand to have a notion of the "grouped" concept.
        # Here we reduce the input/output tensor from 5D to 4D tensor for such cases, e.g.:
        # input operand with dimensionality_order = ['B', 'G', 'C', 'IY', 'IX']
        #   -> gets reduced to dimensionality_order = ['B', 'CH', 'IY', 'IX']
        #       (in this case the 'CH' represents the absolute "channel" dimension)
        node_tensor = node.reshape_operand_tensor(node_tensor, operand=op)

        return node_tensor

    def get_node_tensors(self, node: ComputationNode, tiles: list[ComputationNode]) -> dict[LayerOperand, NodeTensor]:
        variable_operands = [op for op in node.input_operands if op not in node.constant_operands] + [
            node.output_operand
        ]
        tensors_cns: dict[LayerOperand, NodeTensor] = {}
        for op in variable_operands:
            tensors_cns[op] = self.get_node_tensor(node, tiles, op)
        return tensors_cns

    @staticmethod
    def set_base_priority_of_nodes(workload: ComputationNodeWorkload):
        """Set the base_priority of all stored tensors of the variable operands of all nodes
        based on the amount of real (excluding same layer edges) edges.

        Args:
            workload (ComputationNodeWorkload): The workload DiGraph
        """
        for node in workload.node_list:
            output_operand = node.output_operand
            output_tensor = node.operand_tensors[output_operand]
            successors = list(workload.successors(node))
            output_tensor.set_base_priorities(len(successors))

    @staticmethod
    def set_nb_real_predecessors(workload: ComputationNodeWorkload):
        """Set the number of real predecessors for each node in the graph.
        A real predecessor is a node that is not in the same layer as the node itself.
        """
        for node in workload.node_list:
            real_predecessors = [pred for pred in workload.predecessors(node) if pred.id != node.id]
            node.nb_real_predecessors = len(real_predecessors)

    def remake_workload(self, tiles: list[ComputationNode], edges: list[EDGE_T]) -> ComputationNodeWorkload:
        """Remake the workload to fix graph inconsistencies."""
        new_workload = ComputationNodeWorkload()
        new_workload.add_nodes_from(tiles)
        new_workload.add_edges_from(edges)
        return new_workload

    def load_cached_tiled_workload(self) -> ComputationNodeWorkload | None:
        if os.path.exists(self.tiled_workload_path):
            return pickle_load(self.tiled_workload_path)
        return None
