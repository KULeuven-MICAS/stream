from copy import deepcopy
import itertools
from math import ceil, prod
from re import L
from typing import Any, List
import networkx as nx
from networkx import DiGraph
import numpy as np
from rtree import index
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.opt.splitting.TemporalLoop import TemporalLoop
from stream.classes.workload.node import Node
from zigzag.workload.ONNXWorkload import ONNXWorkload as Workload
from zigzag.datatypes import Constants, LayerDim, LayerOperand
from zigzag.utils import pickle_deepcopy
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.lpnormalization_node import LpNormalizationNode
from stream.classes.workload.reshape_node import ReshapeNode
from stream.classes.workload.transpose_node import TransposeNode
from stream.classes.workload.tensor import Tensor
from stream.classes.workload.computation_node import ComputationNode, LoopRanges
from zigzag.stages.Stage import Stage, StageCallable
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.opt.splitting.splitting import (
    convert_inner_cn_loops,
    convert_outer_cn_loops,
    convert_outer_cn_loops_with_k,
)

import logging


logger = logging.getLogger(__name__)


class GenerateCNWorkloadHybridStage(Stage):
    """
    Class that transforms the layer-by-layer workload into finer CN workload graph.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: Workload,
        accelerator: Accelerator,
        cn_define_mode: int,
        hint_loops: list[tuple[LayerDim, str | int]],
        **kwargs: Any,
    ):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator

        # Save for each of the workload's nodes the finer nodes that will be generated
        self.finer_nodes_dict: dict[ComputationNode, list[ComputationNode]] = {}

        # Memoize the numpy tensors for dependency generation
        self.numpy_tensors = {}

        # for CN node size case study, will be used in the function of get_outer_tmap_loop_dimensions
        if cn_define_mode not in [1, 2, 3, 4]:
            raise ValueError(f"cn_define_mode can not be {self.cn_define_mode}.")
        self.cn_define_mode = cn_define_mode
        self.hint_loops = hint_loops  # can be outer-cn or inner-cn depending on cn_define_mode

        # compute the weight capacities of the different cores and the number of splits required for each layer
        if cn_define_mode == 4:
            try:
                self.split_W_percentage = self.kwargs["split_W_percentage"]
            except:
                self.split_W_percentage = 1
            # compute the on-chip weight capacities of the different cores (assumes 'I2' is for weights)
            self.weight_capacities = self.get_weight_capacities()
            # compute the number of splits required for each layer in the original workload
            self.layer_split_factors_k = self.get_layer_split_factors_k()

    def run(self):
        unique_finer_nodes: list[ComputationNode] = []
        # For each node get all the finer nodes and set the intra edges
        G = Workload()
        for node in self.workload.topological_sort():
            # If other node types shouldn't be included in finer node graph, add here
            if not isinstance(node, ComputationNode):
                continue
            outer_temporal_loops = self.get_outer_tmap_loop_dimensions(node)
            finer_nodes, unique_nodes = self.get_finer_nodes(node, outer_temporal_loops)
            logger.info(f"{node}: Outer loops {outer_temporal_loops}.")
            logger.info(f"{node}: Generated {len(finer_nodes)} finer nodes.")
            self.finer_nodes_dict[node] = finer_nodes
            unique_finer_nodes += unique_nodes
            # Compute the edges between nodes originating from one bigger node (intra-edges)
            intra_edges = self.get_intra_edges(finer_nodes)
            G.add_edges_from(intra_edges)
            # If there is only one finer node for this layer, add the node to the graph
            if not intra_edges:
                G.add_nodes_from(finer_nodes)

        # Get all pairs of nodes that we have to extract inter edges for
        all_pairs = self.get_all_node_pairs(self.workload)
        for producer, consumer, is_complex in all_pairs:
            finer_producers = self.finer_nodes_dict[producer]
            finer_consumers = self.finer_nodes_dict[consumer]
            if is_complex:
                inter_edges = self.get_inter_edges_numpy(producer, consumer, finer_producers, finer_consumers)
            else:
                inter_edges = self.get_inter_edges_rtree(producer, consumer, finer_producers, finer_consumers)
            G.add_edges_from(inter_edges)

        # Set the base_priority value of all nodes in G
        self.set_base_priority_of_nodes(G, self.finer_nodes_dict)

        # Set nb of real predecessors of all nodes in G
        self.set_nb_real_predecessors(G)

        logger.info(f"Finer graph: {G}.")

        kwargs = self.kwargs.copy()
        kwargs["original_workload"] = pickle_deepcopy(self.workload)
        kwargs["workload"] = G
        kwargs["accelerator"] = self.accelerator
        kwargs["hint_loops"] = self.hint_loops
        if "scheduling_order" not in kwargs:
            kwargs["scheduling_order"] = self.get_scheduling_order(G)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

        yield None, None

    @staticmethod
    def get_scheduling_order(workload: Workload):
        return sorted(((n.id, n.sub_id) for n in workload.nodes()), reverse=True)

    @staticmethod
    def get_all_node_pairs(G: Workload) -> tuple[tuple[ComputationNode, ComputationNode, bool], ...]:
        pairs: list[tuple[ComputationNode, ComputationNode, bool]] = []
        for node in G.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            successors = list(G.successors(node))
            is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]
            while not all(is_computation_node):
                non_computation_node_succ_idx = is_computation_node.index(False)
                non_computation_node_succ = successors[non_computation_node_succ_idx]
                succ2 = list(G.successors(non_computation_node_succ))
                successors.pop(non_computation_node_succ_idx)
                successors += succ2
                is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]

            # Now we have all ComputationNode successors
            for successor in successors:
                intermediates = nx.shortest_path(G, node, successor)[1:-1]
                complex_pair = False
                for intermediate in intermediates:
                    if not isinstance(intermediate, DummyNode):
                        complex_pair = True
                pairs.append((node, successor, complex_pair))
        return tuple(pairs)

    def get_outer_tmap_loop_dimensions(self, layer: ComputationNode) -> List[TemporalLoop]:
        """Get the temporal loops that are outside a CN for this layer.

        Args:
            layer (Node): layer node for which to return outer-cn loops

        Returns:
            List[TemporalLoop]: list of temporal loops outside of cn
        """
        outer_loops = []
        if self.cn_define_mode == 1:
            # outer_cn_loops identical for all layers
            outer_cn_loops = self.hint_loops.copy()
            outer_loops = convert_outer_cn_loops(outer_cn_loops, layer)
        elif self.cn_define_mode == 2:
            inner_cn_loops = self.hint_loops.copy()
            outer_loops = convert_inner_cn_loops(inner_cn_loops, layer)
        elif self.cn_define_mode == 3:
            # Assume that self.hint_loops is a dict
            # A key is a tuple containing the layer ids that should use the value as hint_loops
            # So for self.hint_loops = {(0,1,2,3): [("OY", "all")], (4,): [("OY", "all"), ("K", "all")]}
            # layer ids 0 to 3 will use [("OY", "all")] and layer id 4 will use [("OY", "all), ("K", "all")]
            # Find which sublist this layer should use
            try:
                outer_cn_loops = next(v for k, v in self.hint_loops.items() if layer.id in k)
            except StopIteration:
                raise ValueError(f"Layer id {layer.id} not in hint_loops: {self.hint_loops}")
            outer_loops = convert_outer_cn_loops(outer_cn_loops, layer)
        elif self.cn_define_mode == 4:
            # Assume we always split in the hint_loops dimensions
            # Check if we need to split in K dimension for it to not block offchip during computation
            outer_cn_loops = self.hint_loops.copy()
            try:
                split_factor = self.layer_split_factors_k[layer]
            except KeyError:
                split_factor = 1
            outer_loops = convert_outer_cn_loops_with_k(outer_cn_loops, layer, split_factor)
        else:
            raise ValueError("This shouldn't be reached if initialization checks are correctly implemented.")
        if not outer_loops:
            outer_loops.append(TemporalLoop(layer.layer_dims[0], 1))
        return outer_loops

    def get_non_type_predecessors(self, node: Node, types: list[type]) -> list[Node]:
        """Find all self.workload nodes that are not of any type in types.
        If a node of any type in types is a predecessor, we cascade back through the graph until only non-types type preds are found.

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

    @staticmethod
    def get_group_id(node: ComputationNode, loop_ranges: LoopRanges, groups: dict) -> tuple[int, dict]:
        """Return the group id for the given loop ranges.
        The group id is determined based on the relevant constant operand dimension loop ranges.
        If there is no constant operand, we return 0.
        If there is more than one constant operand, we only consider the last one's loop ranges.
        If those loop ranges are already contained within 'groups' we return that group id.
        Else we add it to the groups dict with an incremented group id.

        Args:
            node (ComputationNode): The original (layer) CN.
            loop_ranges (dict): A dictionary containing the loop range for each dimension
            groups (dict): The group ids for the already seen loop ranges.

        Returns:
            int: The group id for the given loop ranges
        """
        constant_operands = node.constant_operands
        if not constant_operands:
            return 0, groups
        constant_operand = constant_operands[-1]
        # relevant_dims = node.operand_loop_dim[constant_operand]["r"]
        relevant_dims = node.loop_relevancy_info.get_r_layer_dims(constant_operand)
        relevant_ranges = tuple([loop_ranges[dim] for dim in relevant_dims])
        if relevant_ranges in groups:
            return groups[relevant_ranges], groups
        max_group_id = max(groups.values(), default=-1)
        new_group_id = max_group_id + 1
        groups[relevant_ranges] = new_group_id
        return new_group_id, groups

    @staticmethod
    def get_finer_nodes(
        original_node: ComputationNode, outer_temporal_loops: list[TemporalLoop]
    ) -> tuple[list[ComputationNode], list[ComputationNode]]:
        # Extract the original node id. This should be a tuple of length one.
        # The finer nodes we generate will have a tuple of length two, of format (original_node_id, finer_node_id)
        original_node_id = original_node.id

        # Take away the outer_temporal_loops to create finer CNs for this node
        finer_node_attrs = original_node.extract_node_attr()
        for outer_tl in outer_temporal_loops:
            outer_dim = outer_tl.dimension
            outer_size = outer_tl.size
            # Check if this node's "dim" size is divisible by the outer-cn loop size
            node_dim_size = finer_node_attrs.layer_dim_sizes[outer_dim]
            q, rem = divmod(node_dim_size, outer_size)  # returns x//y, x%y
            assert (
                rem == 0
            ), f"Node {original_node} dim {outer_dim} of size {node_dim_size} is not divisible by outer-cn temporal loop {outer_tl}"
            finer_node_attrs.layer_dim_sizes[outer_dim] = q

        # Loop dimension + size of the finer nodes (called span here)
        finer_span = finer_node_attrs.layer_dim_sizes

        # Get all loop dimensions that the original node has
        loop_dims = original_node.layer_dims

        # Stop value of the outer-cn loops
        stop_values = [temporal_loop.size for temporal_loop in outer_temporal_loops]

        # Number of cns there will be
        nb_cns = int(prod(stop_values))

        # Compute the data_reuse_factor (will be used as base_priority later) for the constant operands of all CNs
        tensor_reuse_factors = deduce_tensor_reuse_factors(original_node, outer_temporal_loops)

        # Multiplication factor for each outer-cn loop.
        # This is to convert from the relative loop value which goes from 0, 1, ..., stop_value - 1
        # to the absolute value of that dimension (if there is another lower loop of the same type or spatial loop)
        mult_factors: list[int] = []
        for i, outer_loop in enumerate(outer_temporal_loops):
            loop_dim = outer_loop.dimension
            stop_value = outer_loop.size
            inner_span = finer_span[loop_dim] if loop_dim in finer_span else 1
            lower_outer_cn_loops = outer_temporal_loops[:i]
            # Returns 1 if empty list
            outer_span = prod(
                [temporal_loop.size for temporal_loop in lower_outer_cn_loops if temporal_loop.dimension == loop_dim]
            )
            mult_factors.append(int(inner_span * outer_span))

        finer_nodes: list[ComputationNode] = []
        groups = {}
        tensors = []
        for n in range(nb_cns):
            outer_loop_values: list[int] = []
            for i, outer_loop in enumerate(outer_temporal_loops):
                loop_dim = outer_loop.dimension
                stop_value = outer_loop.size
                m = prod(stop_values[:i])
                outer_loop_values.append(int((n // m) % stop_value))
            dim_min_max: LoopRanges = {}
            for loop_dim in loop_dims:
                # find all outer-cn loops that iterate over this loop_dim
                # and multiply their loop values by their mult_factor
                dim_min = 0
                for i, outer_loop in enumerate(outer_temporal_loops):
                    dim = outer_loop.dimension
                    stop_value = outer_loop.size
                    if dim == loop_dim:
                        # current loop value of this outer-cn loop
                        loop_val = outer_loop_values[i]
                        # mult factor of this outer-cn loop
                        mult_factor = mult_factors[i]
                        dim_min += loop_val * mult_factor
                # max value is exclusive
                dim_max = dim_min + (finer_span[loop_dim] if loop_dim in finer_span else 1)
                dim_min_max[loop_dim] = (dim_min, dim_max)

            # Add the loop ranges for this cn to a copy of the finer node attributes
            finer_node_attrs_copy = deepcopy(finer_node_attrs)

            # Determine the group id of this layer based on the loop ranges
            group_id, groups = GenerateCNWorkloadHybridStage.get_group_id(original_node, dim_min_max, groups)

            # Create the computation node object with the computed ranges of the loop dimensions
            node_name = original_node.name
            node_input_names = original_node.input_names
            node_output_names = original_node.output_names
            # If all the output irrelevant loops are at a max, this is producing a final output, so set a flag
            original_node_output_ir_dims = original_node.loop_relevancy_info.get_ir_layer_dims(
                Constants.OUTPUT_LAYER_OP
            )

            produces_final_output = all(
                [dim_min_max[dim][1] >= original_node.layer_dim_sizes[dim] for dim in original_node_output_ir_dims]
            )

            finer_node = ComputationNode(
                node_id=original_node_id,
                sub_id=n,
                node_name=node_name,
                node_attr=finer_node_attrs_copy,
                input_names=node_input_names,
                output_names=node_output_names,
                op_type=original_node.type,
                produces_final_output=produces_final_output,
                group_id=group_id,
            )
            # Override loop_ranges property
            finer_node.update_loop_ranges(dim_min_max)
            # Re-calculate pr loop ranges based on new loop_ranges
            finer_node.calculate_pr_loop_ranges()
            # Re-set the operand tensors for the new loop_ranges
            finer_node.set_operand_tensors()

            # Initialize the priorities (total inter-CN data reuse factor) for the constant operands of this finer_node
            for constant_operand in finer_node.constant_operands:
                tensor = finer_node.operand_tensors[constant_operand]
                tensor.set_base_priorities(tensor_reuse_factors[constant_operand][n])

            # Replace any of the tensors with identical tensors of previous finer nodes
            for op, tensor in finer_node.operand_tensors.items():
                replaced = False
                for previous_tensor in tensors:
                    if tensor.equality_hash() == previous_tensor.equality_hash():
                        finer_node.operand_tensors[op] = previous_tensor
                        replaced = True
                if not replaced:
                    tensors.append(tensor)

            # Compute the output data produced by each finer node, assuming that all the data produced by different CNs are unique
            finer_node.data_produced_unique = (
                finer_node.operand_size_elem[Constants.OUTPUT_LAYER_OP]
                * finer_node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            )

            # TODO Compute the unique input data consumed by each finer node. Note that it is not necessarily that the data consumed by different CNs are unique
            # finer_node.data_consumed_unique = ... (for now it is 0)

            finer_nodes.append(finer_node)

        # TODO Just take the first node as they are all equal for now. If some are different, this should be done more smartly
        unique_finer_nodes = [finer_nodes[0]]

        return finer_nodes, unique_finer_nodes

    @staticmethod
    def get_intra_edges(nodes: list[ComputationNode]):
        # Get all the group ids
        group_ids = sorted(set([n.group for n in nodes]))
        intra_edges: list[tuple[ComputationNode, ComputationNode, dict[str, int]]] = []
        for group_id in group_ids:
            group_nodes = [n for n in nodes if n.group == group_id]
            pairs = zip(group_nodes, group_nodes[1:])
            for node_1, node_2 in pairs:
                intra_edges.append((node_1, node_2, {"bits": 0}))
        return intra_edges

    def convert_to_inclusive_data_range(self, exclusive_data_range: LoopRanges):
        """
        Convert an exclusive data range to an inclusive data range.
        """
        return {key: (min_val, max_val - 1) for key, (min_val, max_val) in exclusive_data_range.items()}

    def get_bounding_box_dimensions(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
        dimensions: list[LayerDim],
        loop_ranges: LoopRanges,
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

        if not interleaved:
            bounding_box_flat = tuple([item for sublist in bounding_box for item in sublist])
            return bounding_box_flat
        else:
            bounding_box_flat = tuple(zip(*bounding_box))
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
            yield (i, bounds, None)

    def get_nb_input_dimensions(self, node: ComputationNode):
        """Return the number of input dimensions this node has. We take the first non-constant input operand."""
        input_operand = list(set(node.input_operands) - set(node.constant_operands))[0]
        dims = node.operand_dimensionality_order[input_operand]
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
        props.dimension = self.get_nb_input_dimensions(nodes[0])

        rtree = index.Index(self.bounding_box_generator(producer, consumer, nodes, operand), properties=props)
        return rtree

    def flatten_grouped_convolution_ranges(
        self, producer: ComputationNode, consumer: ComputationNode, dims: list[LayerDim], ranges: LoopRanges
    ):
        """If both C/K and G are present in dimensions, flatten their loop ranges so the tensor is 4D.

        Args:
            dimensions (list): list of the different tensor dimensions
            loop_ranges (dict): dict of the loop ranges for the current node.
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
        finer_producers: list[ComputationNode],
        finer_consumers: list[ComputationNode],
    ):
        """Function that finds the edges between a producer and consumer node,
        more specifically their finer counterparts producer_finer and consumer_finer.
        A communication node is inserted between each producer and consumer node.

        Args:
            producer (Node): the producer node
            consumer (Node): the consumer node
            finer_producers (list): list of finer producer nodes
            finer_consumers (list): list of finer consumer nodes
        """
        # _dims = self.get_nb_input_dimensions(finer_consumers[0])
        # assert all(
        #     [
        #         len(_producer.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]) == _dims
        #         for _producer in finer_producers
        #     ]
        # )

        # Check all the different input operands of the consumer node that stem from the producer node
        # The direct predecessor of an input operand might be a DummyNode so we need to propagate back
        dependent_input_operands: list[LayerOperand] = []
        for operand, parent_node_id in consumer.input_operand_source.items():
            parent_node = self.workload.get_node_with_id(parent_node_id)
            assert isinstance(parent_node, Node)
            if parent_node == producer:
                dependent_input_operands.append(operand)
            elif parent_node:
                non_dummy_parents = self.get_non_type_predecessors(parent_node, [DummyNode])
                if producer in non_dummy_parents:
                    dependent_input_operands.append(operand)

        # edges will hold the cns that are dependent on each other [(prod_cn, cons_cn), ...]
        edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []

        for input_operand in dependent_input_operands:
            # Build the tree of all finer consumer nodes for this operand
            consumer_tree = self.build_rtree(producer, consumer, finer_consumers, input_operand)

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
            for finer_producer in finer_producers:
                # Get the output irrelevant loop ranges and check if they are at least at the max
                ir_dims_not_at_max = [
                    finer_producer.loop_ranges[ir_dim][1] < producer.loop_ranges[ir_dim][1]
                    for ir_dim in producer_ir_dims_output
                ]
                if any(ir_dims_not_at_max):
                    continue  # to the next finer producer

                p_inclusive_ranges = self.convert_to_inclusive_data_range(finer_producer.loop_ranges)
                p_bounding_box = self.get_bounding_box_dimensions(
                    producer, consumer, producer_r_dims_output, p_inclusive_ranges
                )

                # Get the finer consumer node ids that intersect with this finer producer node
                intersecting_consumer_node_ids = consumer_tree.intersection(p_bounding_box)

                for intersecting_consumer_node_id in intersecting_consumer_node_ids:
                    intersecting_consumer = finer_consumers[intersecting_consumer_node_id]
                    # Create a new communication node that will reside between the producer and consumer node
                    edges += [
                        (
                            finer_producer,
                            intersecting_consumer,
                            {
                                "operand": input_operand,
                                "bits": finer_producer.data_produced_unique,
                            },
                        )
                    ]

        return edges

    def get_inter_edges_numpy(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
        finer_producers: list[ComputationNode],
        finer_consumers: list[ComputationNode],
    ):
        numpy_tensors: dict[ComputationNode, dict] = {}
        # Get the paths from producer to consumer
        paths_between_generator = nx.all_simple_paths(self.workload, source=producer, target=consumer)
        all_inter_edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []
        for path_between in paths_between_generator:
            dependent_operand = Constants.OUTPUT_LAYER_OP
            # FIRST NODE
            # First node in the path is a ComputationNode, of which we extract the output operand dependency tensor
            node = path_between[0]
            assert isinstance(node, ComputationNode), "First node in path should be ComputationNode"
            if node in numpy_tensors:
                tensor_cns = numpy_tensors[node]
            else:
                finer_nodes = self.finer_nodes_dict[node]
                tensor_cns = self.get_tensor_cns(node, finer_nodes)
                numpy_tensors[node] = tensor_cns
                tensor = tensor_cns[Constants.OUTPUT_LAYER_OP]
            # INTERMEDIATE NON-COMPUTATION NODES
            for _, node in enumerate(path_between[1:-1], start=1):
                if isinstance(node, ComputationNode):
                    raise ValueError("Intermediate nodes should not be of type ComputationNode.")
                tensor = self.propagate_cn_production_for_non_cn(node, tensor)
            # LAST NODE IN PATH
            last_node: Node = path_between[-1]
            # Find the operand for which this last node connects to its predecessor

            dependent_operand = next(
                op for op, dependent_node_id in last_node.input_operand_source.items() if dependent_node_id == node.id
            )
            if last_node in numpy_tensors:
                tensor_cns = numpy_tensors[last_node]
            else:
                finer_nodes = self.finer_nodes_dict[last_node]
                tensor_cns = self.get_tensor_cns(last_node, finer_nodes)
                numpy_tensors[node] = tensor_cns
            last_tensor = tensor_cns[dependent_operand]
            inter_edges = self.get_inter_edges_tensor_based(tensor, last_tensor)
            for prod, cons in inter_edges:
                all_inter_edges.append(
                    (
                        prod,
                        cons,
                        {
                            "operand": dependent_operand,
                            "bits": prod.data_produced_unique,
                        },
                    )
                )
        return all_inter_edges

    def propagate_cn_production_for_non_cn(
        self, node: Node, input_tensor: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        if isinstance(node, ReshapeNode):
            output_tensor = node.reshape_operand_tensor(input_tensor)
        elif isinstance(node, TransposeNode):
            output_tensor = node.transpose(input_tensor)
        elif isinstance(node, LpNormalizationNode):
            output_tensor = node.lpnormalization_operand_tensor(input_tensor)
        elif isinstance(node, FlattenNode):
            output_tensor = node.flatten(input_tensor)
        elif isinstance(node, ElementwiseNode):
            output_tensor = input_tensor.copy()
        else:
            raise NotImplementedError(f"Tensor propagation not implemented for node {node.name}.")
        return output_tensor

    @staticmethod
    def get_inter_edges_tensor_based(
        producer_output_tensor: np.ndarray[Any, Any], consumer_input_tensor: np.ndarray[Any, Any]
    ):
        """This method obtains the edges between a producer and consumer.
        This is done by iterating through all finer consumer nodes,
        for each consumer node we create a window and get all the producer nodes that produced this data window.

        Args:
            producer_output_tensor (np.ndarray): A tensor containing for each position which CNs will produce it
            consumer_input_tensor (np.ndarray): A tensor containing for each position which CNs will consume it
        """
        assert (
            producer_output_tensor.shape == consumer_input_tensor.shape
        ), "Arrays to construct inter-layer edges must be equal shape."
        inter_edges: set[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = set()
        for producer_set, consumer_set in zip(producer_output_tensor.flat, consumer_input_tensor.flat):
            if consumer_set is None:  # Happens for downsample layers (e.g. ComputationNode((16,)) for MBNetV2)
                continue
            for prod, cons in itertools.product(producer_set, consumer_set):
                inter_edges.add((prod, cons))
        return inter_edges

    def get_tensor_cns(
        self, node: ComputationNode, finer_nodes: list[ComputationNode]
    ) -> dict[LayerOperand, np.ndarray[Any, Any]]:
        is_source_node = len(self.get_non_type_predecessors(node, [DummyNode])) == 0
        variable_operands = [op for op in node.input_operands if op not in node.constant_operands] + [
            node.output_operand
        ]
        tensor_dims = {op: node.operand_dimensionality_order[op] for op in variable_operands}
        all_loop_dim_sizes = node.layer_dim_sizes + node.pr_layer_dim_sizes  # union
        tensor_shapes = {op: tuple([all_loop_dim_sizes[dim] for dim in dims]) for (op, dims) in tensor_dims.items()}
        tensors_cns: dict[LayerOperand, np.ndarray[Any, Any]] = {
            op: np.ndarray(shape, dtype=set) for (op, shape) in tensor_shapes.items()
        }  # Initial arrays
        # Fill the initial arrays with an empty set in each position
        for op in variable_operands:
            shape = tensor_shapes[op]
            for idx in itertools.product(*(range(s) for s in shape)):
                tensors_cns[op][idx] = set()

        # For each input operand iterate through the finer_nodes in reverse order
        # because we want the first cn with a dependency saved in the tensor
        # For the output operand iterate through the finer_nodes in regular order
        # because we want the last CN that handles an output tensor window to be saved
        for op in variable_operands:
            dims = tensor_dims[op]
            if op == node.output_operand:
                ir_dims_output = node.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
                finer_nodes_list = finer_nodes  # list in regular order
                should_add_to_tensor_list = [
                    all([finer_node.loop_ranges[ir_dim][1] >= node.loop_ranges[ir_dim][1] for ir_dim in ir_dims_output])
                    for finer_node in finer_nodes_list
                ]
                attr_to_add_to = "data_produced_unique"
                precision = node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            else:
                finer_nodes_list = list(reversed(finer_nodes))  # list in reversed order
                should_add_to_tensor_list = [True for _ in finer_nodes_list]
                attr_to_add_to = "data_consumed_unique"
                precision = node.operand_precision[op] * (
                    not is_source_node
                )  # if this layer is the first layer, we assume the inputs are streamed and "free"
            nb_unique_data_seen = 0
            for finer_node, should_add_to_tensor in zip(finer_nodes_list, should_add_to_tensor_list):
                if not should_add_to_tensor:
                    continue  # Skip if we're not at the max ir loop value for output
                op_dim_ranges = [finer_node.loop_ranges[loop_dim] for loop_dim in dims]
                op_dim_ranges_max_stop = tuple(tensor_shapes[op])
                window = tuple(
                    [slice(max(0, start), stop) for (start, stop) in op_dim_ranges]
                )  # start can be negative for padding which, makes np flip
                # Count how many nans we have in this window, as this is the amount of unique data consumed/produced by this finer_node
                nb_unique_data_bits = np.sum(tensors_cns[op][window] == set()) * precision
                nb_unique_data_seen += nb_unique_data_bits
                # Add this amount of unique data to the "data_consumed_unique" or "data_produced_unique" depending on input/output operand
                setattr(
                    finer_node,
                    attr_to_add_to,
                    getattr(finer_node, attr_to_add_to) + nb_unique_data_bits,
                )
                # Set this window of the tensor to indicate it will be consumed/produced by this finer node
                bounded_op_dim_ranges = [
                    range(max(0, start), min(max_stop, stop))
                    for ((start, stop), max_stop) in zip(op_dim_ranges, op_dim_ranges_max_stop)
                ]
                for index in itertools.product(*bounded_op_dim_ranges):
                    tensors_cns[op][index] |= {finer_node}  # Union of the existing set with the newly added node
            if nb_unique_data_seen != (prod(tensor_shapes[op]) * precision):
                logger.warn(f"Downsampling node detected: {node}, operand= {op}.")

        # The dimensionality order of this input/output operand might include
        # both a G and C/K dimension because the ComputationNode gets the group as an extra
        # dimension in its input/output operand to have a notion of the "grouped" concept.
        # Here we reduce the input/output tensor from 5D to 4D tensor for such cases, e.g.:
        # input operand with dimensionality_order = ['B', 'G', 'C', 'IY', 'IX']
        #   -> gets reduced to dimensionality_order = ['B', 'CH', 'IY', 'IX']
        #       (in this case the 'CH' represents the absolute "channel" dimension)
        for op, tensor in tensors_cns.items():
            tensors_cns[op] = node.reshape_operand_tensor(tensor, operand=op)

        return tensors_cns

    @staticmethod
    def set_base_priority_of_nodes(G: Workload, finer_nodes_dict: dict[ComputationNode, list[ComputationNode]]):
        """Set the base_priority of all stored tensors of variable operands in every node in finer_nodes
         based on the amount of real (excluding same layer edges) edges.

        Args:
            finer_nodes (list): List of the nodes for which to set the tensors' base_priority
        """
        nb_nodes_per_layer_id = {layer.id: len(finer_nodes_dict[layer]) for layer in finer_nodes_dict.keys()}
        nb_seen_nodes_per_layer_id = {layer_id: 0 for layer_id in nb_nodes_per_layer_id.keys()}
        for node in G.topological_sort():
            layer_id = node.id
            for layer_operand in node.layer_operands:
                tensor: Tensor = node.operand_tensors[layer_operand]
                if layer_operand == node.output_operand:
                    # Look at the amount of successors from different layers
                    successors = [succ for succ in G.successors(node) if succ.id != layer_id]
                    tensor.set_base_priorities(len(successors))
            nb_seen_nodes_per_layer_id[layer_id] += 1

    def set_nb_real_predecessors(self, G: Workload):
        """Set nb_real_predecessors attribute for each node in G.
        A real predecessor is a predecessor coming from a different layer.

        Args:
            G (DiGraph): Graph containing the nodes and edges.
        """
        for n in G.nodes():
            nb_real_predecessors = len(list(pred for pred in G.predecessors(n) if pred.id != n.id))
            n.set_nb_real_predecessors(nb_real_predecessors)

    def get_weight_capacities(self):
        # Get the weight capacity of all cores
        weight_capacities = {}
        for core in self.accelerator.cores.nodes():
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(Constants.MEM_OP_2).memory_instance.size
            weight_capacities[core.id] = core_weight_capacity
        return weight_capacities

    def get_layer_split_factors_k(self):
        # Get for each layer the split factor we need to be able to fit weights on possible cores
        split_factors = {}
        for node in self.workload.nodes():
            # Get the weight capacity of all possible core allocations of this node
            core_allocations = node.possible_core_allocation
            if isinstance(node, DummyNode):
                continue
            # for fixed single allocation don't consider the splitting
            if len(core_allocations) == 1:
                continue
            core_capacities = [self.weight_capacities[core_id] for core_id in core_allocations]
            min_core_capacity = min(core_capacities)
            # Get the weight size of this layer
            constant_operands = node.constant_operands
            if not constant_operands:
                continue

            constant_operand = node.constant_operands[0]
            # if "W" in node.constant_operands:
            #     constant_operand = "W"
            # elif "B" in node.constant_operands:
            #     constant_operand = "B"
            # else:
            #     raise NotImplementedError(
            #         f"Layer splitting not implemented for {node} with constant operands= {node.constant_operands}."
            #     )

            weight_size = node.operand_size_bit[constant_operand]
            if weight_size == 0:
                continue
            split_factor = ceil(weight_size / (self.split_W_percentage * min_core_capacity))  # 0.5 for double buffering
            if split_factor == 1:
                continue
            # Check if the split_factor is a divisor of the number of output channels
            try:
                output_channels = node.layer_dim_sizes[LayerDim("K")]
            except KeyError:
                raise NotImplementedError(f"{node} doesn't have a 'K' loop.")
            while divmod(output_channels, split_factor)[1] != 0:
                split_factor += 1
                if split_factor > output_channels:
                    raise ValueError("Something went wrong.")
            split_factors[node] = split_factor
        return split_factors


def deduce_tensor_reuse_factors(
    original_node: ComputationNode, outer_temporal_loops: list[TemporalLoop]
) -> dict[LayerOperand, list[int]]:
    """This function is used to generate a list of inter-CN data reuse factor for each CN's constant operand, like W, based on the outer-CN loops and the r, ir relations.

    Args:
        original_node (ComputationNode): the original layer node before tilling
        outer_temporal_loops (list[TemporalLoop]): the outer CN temporal loops

    Returns:
        data_reuse_factor (dict[list[int]]): a list of data reuse factor (base priority) for constant operands of each CN
    """

    # If there is no loop in the r_ir_loop, meaning that there is no outer-CN loop -> layer-by-layer
    if not outer_temporal_loops:
        return {}

    if not original_node.constant_operands:
        return {}

    # Transfer the outer_temporal_loops to r_ir_loop. An example can be r_ir_loop = {'W': [('ir', 3), ('r', 2), ('ir', 3)]}.
    r_ir_LUT = original_node.loop_relevancy_info
    constant_operands = original_node.constant_operands
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
    # whenveer a cn will be scheduled, the tensor reuse factor will decrease
    tensor_reuse_factors: dict[LayerOperand, list[int]] = {}
    for op, total_reuse_factor in total_reuse_factors.items():
        tensor_reuse_factors[op] = [total_reuse_factor] * nb_nodes

    return tensor_reuse_factors
