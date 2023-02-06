import itertools
from math import prod
from re import L
from typing import List, Dict
import networkx as nx
import numpy as np
from rtree import index
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.lpnormalization_node import LpNormalizationNode
from stream.classes.workload.reshape_node import ReshapeNode
from stream.classes.workload.transpose_node import TransposeNode
from stream.classes.workload.tensor import Tensor

from zigzag.classes.mapping.temporal.temporal_loop import TemporalLoop
from stream.classes.workload.communication_node import CommunicationNode
from stream.classes.workload.computation_node import ComputationNode
from zigzag.classes.stages.Stage import Stage
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.workload.pooling_node import PoolingNode
from zigzag.utils import pickle_deepcopy

import logging
logger = logging.getLogger(__name__)

class GenerateCNWorkloadHybridStage(Stage):
    """
    Class that transforms the layer-by-layer workload into finer CN workload graph.
    """
    def __init__(self, list_of_callables, *, workload, accelerator, cn_define_mode, hint_loops, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.cores = accelerator.cores
        self.core_ids = [core.id for core in self.cores]

        # Save for each of the workload's nodes the finer nodes that will be generated
        self.finer_nodes_dict = {}

        # Memoize the numpy tensors for dependency generation
        self.numpy_tensors = {}

        # for CN node size case study, will be used in the function of get_outer_tmap_loop_dimensions
        self.cn_define_mode = cn_define_mode
        if cn_define_mode == 1:
            self.outer_CN_loops = hint_loops
        elif cn_define_mode == 2:
            self.inner_CN_loops = hint_loops
        elif cn_define_mode == 3:
            self.factor_loops = hint_loops
        else:
            raise ValueError(f"CN_define_mode can not be {self.cn_define_mode}.")

    def run(self):
        unique_finer_nodes = []
        # For each node get all the finer nodes and set the intra edges
        G = nx.DiGraph()
        for node in nx.topological_sort(self.workload):
            if not isinstance(node, ComputationNode):  # If other node types shouldn't be included in finer node graph, add here
                continue
            outer_temporal_loops = self.get_outer_tmap_loop_dimensions(node)
            finer_nodes, unique_nodes = self.get_finer_nodes(node, outer_temporal_loops)
            logger.info(f"{node}: Generated {len(finer_nodes)} finer nodes, {len(unique_nodes)} unique nodes based on outer temporal loops {outer_temporal_loops}.")
            self.finer_nodes_dict[node] = finer_nodes
            unique_finer_nodes += unique_nodes
            # Compute the edges between nodes originating from one bigger node (intra-edges)
            intra_edges = self.get_intra_edges(finer_nodes)
            G.add_edges_from(intra_edges)

        # Get all pairs of nodes that we have to extract inter edges for
        all_pairs = self.get_all_node_pairs(self.workload)
        for (producer, consumer, is_complex) in all_pairs:
            finer_producers = self.finer_nodes_dict[producer]
            finer_consumers = self.finer_nodes_dict[consumer]
            # print((producer, consumer, is_complex))
            if is_complex:
                inter_edges = self.get_inter_edges_numpy(producer, consumer, finer_producers, finer_consumers)
            else:
                inter_edges = self.get_inter_edges_rtree(producer, consumer, finer_producers, finer_consumers)
            G.add_edges_from(inter_edges)
            # print(G)

        # Set the base_priority value of all nodes in G
        self.set_base_priority_of_nodes(G, self.finer_nodes_dict)

        # Set nb of real predecessors of all nodes in G
        self.set_nb_real_predecessors(G)

        logger.info(f"Finer graph: {G}.")

        kwargs = self.kwargs.copy()
        kwargs["workload"] = G
        kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

        yield None, None

    @staticmethod
    def get_all_node_pairs(G):
        pairs = []
        for node in nx.topological_sort(G):
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

    @staticmethod
    def get_rest_loops(total_loop_dim: Dict[str, int], to_be_excluded_loops: List[TemporalLoop]) -> List[TemporalLoop]:
        """
        This function return a list of the rest temporal loops after remove the to_be_excluded_loops from the total_loop_dim.
        """
        rest_loops = []
        to_be_excluded_loops = {TM_loop.dimension: TM_loop.size for TM_loop in to_be_excluded_loops}
        for loop_name, loop_value_total in total_loop_dim.items():
            if loop_name in to_be_excluded_loops:
                loop_value_to_be_gone = to_be_excluded_loops[loop_name]
                loop_value_left = loop_value_total // loop_value_to_be_gone
                if loop_value_left > 1:
                    rest_loops.append(TemporalLoop(loop_name, loop_value_left))
            else:
                if loop_value_total > 1:
                    rest_loops.append(TemporalLoop(loop_name, loop_value_total))
        return rest_loops

    @staticmethod
    def find_the_closest_divisible_factor_within_a_range(total, factor, a_range):
        """
        This function find the closest divisible factor within a range.
        E.g., if the total loop size 26, the factor is 10, and the range is 2,
        the function will try all the values between 10/2 and 10*2, and return 13 as result.
        """
        lower_bound = max(2, factor//a_range)
        upper_bound = min(total, factor*a_range)
        new_factor_candidates = [(i, abs(factor-i)) for i in range(lower_bound, upper_bound+1) if total % i == 0]

        new_factor = min(new_factor_candidates, key=lambda tup: tup[1])[0]
        return new_factor

    def get_outer_tmap_loop_dimensions(self, layer) -> List[TemporalLoop]:
        """Get the temporal loops that are outside a CN for this layer.

        Args:
            layer (Node): layer node for which to return outer-cn loops

        Returns:
            List[TemporalLoop]: list of temporal loops outside of cn
        """
        outer_loops = []

        if self.cn_define_mode == 1:
            for (loop_name, loop_size) in self.outer_CN_loops:
                if loop_name in layer.loop_dim_size:
                    if loop_size == 'all' or layer.loop_dim_size[loop_name] < loop_size:
                        outer_loops.append(TemporalLoop(loop_name, layer.loop_dim_size[loop_name]))
                    elif layer.loop_dim_size[loop_name] % loop_size == 0:
                        outer_loops.append(TemporalLoop(loop_name, loop_size))
                    else:
                        try:
                            # find the closest factor within 50x.
                            new_loop_size = self.find_the_closest_divisible_factor_within_a_range(layer.loop_dim_size[loop_name], loop_size, 50)
                            outer_loops.append(TemporalLoop(loop_name, new_loop_size))
                            logger.info(f"For layer {int(layer.id[0])}, the outer CN dimension {loop_name} size is adjusted from {loop_size} to {new_loop_size}.")
                        except:
                            raise ValueError(f"({loop_name}, {loop_size}) is not a valid outer CN loop.")

        elif self.cn_define_mode == 2:
            inner_loops = []
            for (loop_name, loop_size) in self.inner_CN_loops:
                if loop_name in layer.loop_dim_size:
                    if loop_size == 'all' or layer.loop_dim_size[loop_name] < loop_size:
                        inner_loops.append(TemporalLoop(loop_name, layer.loop_dim_size[loop_name]))
                    elif layer.loop_dim_size[loop_name] % loop_size == 0:
                        inner_loops.append(TemporalLoop(loop_name, loop_size))
                    else:
                        try:
                            # find the closest factor within 50x.
                            new_loop_size = self.find_the_closest_divisible_factor_within_a_range(layer.loop_dim_size[loop_name], loop_size, 50)
                            outer_loops.append(TemporalLoop(loop_name, new_loop_size))
                            logger.info(f"For layer {int(layer.id[0])}, the inner CN dimension {loop_name} size is adjusted from {loop_size} to {new_loop_size}.")
                        except:
                            raise ValueError(f"({loop_name}, {loop_size}) is not a valid inner CN loop.")
            outer_loops = self.get_rest_loops(layer.loop_dim_size, inner_loops)

        else:
            # TODO
            inner_loops = []
            outer_loops = []

        if not outer_loops:
            outer_loops.append(TemporalLoop(layer.loop_dim_list[0], 1))
        return outer_loops

    def get_non_type_predecessors(self, node, types):
        """Find all self.workload nodes that are not of any type in types.
        If a node of any type in types is a predecessor, we cascade back through the graph until only non-types type preds are found.

        Args:
            node (Node): the node for which we intend to find all preds that are not of a type in types
            types (list): list of different types that we want to exclude from our predecessors
        """
        preds = list(self.workload.predecessors(node))
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
    def get_finer_nodes(original_node, outer_temporal_loops) -> tuple[list[ComputationNode], list[ComputationNode]]:
        # Extract the original node id. This should be a tuple of length one.
        # The finer nodes we generate will have a tuple of length two, of format (original_node_id, finer_node_id)
        original_node_id = original_node.id[0]

        # Take away the outer_temporal_loops to create finer CNs for this node
        finer_node_attrs = original_node.attrs.copy()
        for outer_tl in outer_temporal_loops:
            outer_dim = outer_tl.dimension
            outer_size = outer_tl.size
            # Check if this node's "dim" size is divisible by the outer-cn loop size
            node_dim_size = finer_node_attrs["loop_dim_size"][outer_dim]
            q, rem = divmod(node_dim_size, outer_size)  # returns x//y, x%y
            assert rem == 0, f"Node {original_node} dim {outer_dim} of size {node_dim_size} is not divisible by outer-cn temporal loop {outer_tl}"
            finer_node_attrs["loop_dim_size"][outer_dim] = q

        # Loop dimension + size of the finer nodes (called span here)
        finer_span = finer_node_attrs["loop_dim_size"]

        # Get all loop dimensions that the original node has
        loop_dims = original_node.loop_dim_list

        # Stop value of the outer-cn loops
        stop_values = [temporal_loop.size for temporal_loop in outer_temporal_loops]

        # Number of cns there will be
        nb_cns = int(prod(stop_values))

        # Compute the data_reuse_factor (will be used as base_priority later) for the constant operands of all CNs
        tensor_reuse_factors = deduce_tensor_reuse_factors(original_node, outer_temporal_loops)

        # Multiplication factor for each outer-cn loop.
        # This is to convert from the relative loop value which goes from 0, 1, ..., stop_value - 1
        # to the absolute value of that dimension (if there is another lower loop of the same type or spatial loop)
        mult_factors = []
        for i, outer_loop in enumerate(outer_temporal_loops):
            loop_dim = outer_loop.dimension
            stop_value = outer_loop.size
            inner_span = finer_span.get(loop_dim, 1)  # Return 1 if loop_dim is not in the cn_span dict
            lower_outer_cn_loops = outer_temporal_loops[:i]
            outer_span = prod([temporal_loop.size for temporal_loop in lower_outer_cn_loops if temporal_loop.dimension == loop_dim])  # Returns 1 if empty list
            mult_factors.append(int(inner_span * outer_span))

        finer_nodes = []
        for n in range(nb_cns):
            outer_loop_values = []
            for i, outer_loop in enumerate(outer_temporal_loops):
                loop_dim = outer_loop.dimension
                stop_value = outer_loop.size
                m = prod(stop_values[:i])
                outer_loop_values.append(int((n//m) % stop_value))
            dim_min_max = {}
            for loop_dim in loop_dims:
                # find all outer-cn loops that iterate over this loop_dim
                # and multiply their loop values by their mult_factor
                dim_min = 0
                for i, outer_loop in enumerate(outer_temporal_loops):
                    dim = outer_loop.dimension
                    stop_value = outer_loop.size
                    if dim == loop_dim:
                        loop_val = outer_loop_values[i]  # current loop value of this outer-cn loop
                        mult_factor = mult_factors[i]  # mult factor of this outer-cn loop
                        dim_min += loop_val * mult_factor
                dim_max = dim_min + finer_span.get(loop_dim, 1)  # max value is exclusive
                dim_min_max[loop_dim] = (dim_min, dim_max)
            
            # Add the loop ranges for this cn to a copy of the finer node attributes
            finer_node_attrs_copy = finer_node_attrs.copy()
            finer_node_attrs_copy["loop_ranges"] = dim_min_max

            # TODO Compute the discarded inputs of each finer node
            # TODO Compute the spawning outputs of each finer node

            # Create the computation node object with the computed ranges of the loop dimensions and number of dying pixels
            node_name = original_node.name
            node_input_names = original_node.input_names
            node_output_names = original_node.output_names
            # If all the output irrelevant loops are at a max, this is producing a final output, so set a flag
            original_node_output_ir_dims = original_node.operand_loop_dim['O']['ir']
            if all([dim_min_max[dim][1] >= original_node.loop_dim_size[dim] for dim in original_node_output_ir_dims]):
                produces_final_output = True
            else:
                produces_final_output = False
            finer_node = ComputationNode((original_node_id, n), finer_node_attrs_copy, node_name, node_input_names, node_output_names, produces_final_output=produces_final_output)

            # Initialize the priorities (total inter-CN data reuse factor) for the constant operands of this finer_node
            for constant_operand in finer_node.constant_operands:
                tensor = finer_node.operand_tensors[constant_operand]
                tensor.set_base_priorities(tensor_reuse_factors[constant_operand][n])

            # Compute the output data produced by each finer node, assuming that all the data produced by different CNs are unique
            finer_node.data_produced_unique = finer_node.operand_size_elem['O'] * finer_node.operand_precision['O_final']

            # TODO Compute the unique input data consumed by each finer node. Note that it is not necessarily that the data consumed by different CNs are unique
            # finer_node.data_consumed_unique = ... (for now it is 0)

            finer_nodes.append(finer_node)

        # TODO Just take the first node as they are all equal for now. If some are different, this should be done more smartly
        unique_finer_nodes = [finer_nodes[0]]

        return finer_nodes, unique_finer_nodes

    @staticmethod
    def get_intra_edges(nodes):
        return [(nodes[node_id], nodes[node_id + 1], {'bits': 0}) for node_id in range(len(nodes) - 1)]

    def get_inter_edges_rtree(self, producer, consumer, finer_producers, finer_consumers):
        """Function that finds the edges between a producer and consumer node,
        more specifically their finer counterparts producer_finer and consumer_finer.
        A communication node is inserted between each producer and consumer node.

        Args:
            producer (Node): the producer node
            consumer (Node): the consumer node
            finer_producers (list): list of finer producer nodes
            finer_consumers (list): list of finer consumer nodes
        """
        def convert_to_inclusive_data_range(exclusive_data_range):
            """
            Convert an exclusive data range to an inclusive data range.
            """
            return {key: (min_val, max_val - 1) for key, (min_val, max_val) in exclusive_data_range.items()}

        def flatten_grouped_convolution_ranges(dims, ranges):
            """If both C/K and G are present in dimensions, flatten their loop ranges so the tensor is 4D.

            Args:
                dimensions (list): list of the different tensor dimensions
                loop_ranges (dict): dict of the loop ranges for the current node.
            """
            dims_copy = dims.copy()
            ranges_copy = ranges.copy()
            if 'G' in dims_copy and ('C' in dims_copy or 'K' in dims_copy):
                G_idx = dims_copy.index('G')
                if 'C' in dims_copy:
                    is_consumer = True
                    C_K_idx = dims_copy.index('C')
                elif 'K' in dims_copy:
                    C_K_idx = dims_copy.index('K')
                    is_consumer = False
                else:
                    return dims_copy, ranges_copy
                # Replace the G + C/K into one dimension we call "CH" (name doesn't really matter)
                (G_min, G_max_incl) = ranges_copy['G']
                (C_K_min, C_K_max_incl) = ranges_copy[dims_copy[C_K_idx]]
                CH_min = G_min + C_K_min
                original_node = consumer if is_consumer else producer
                CH_max_incl = G_max_incl * original_node.loop_dim_size[dims_copy[C_K_idx]] + C_K_max_incl
                ranges_copy["CH"] = (CH_min, CH_max_incl)
                
                # Remove the G + C/K from the original dimensions list and add CH in its place
                min_idx = min(G_idx, C_K_idx)

                dims_copy.remove('G')
                second_dim = 'C' if is_consumer else 'K'
                dims_copy.remove(second_dim)
                dims_copy.insert(min_idx, "CH")  # insert it in place of G or C/K, whichever came first

            return dims_copy, ranges_copy

        def get_bounding_box_dimensions(dimensions, loop_ranges, interleaved=True):
            """
            Extract the relevant dimension ranges for building the rtree with the dimensions in dimensions.
            The order of the operand's dimensions is determined through the dimensions parameter.
            """
            # Add compensation for grouped convolutions:
            # If there is a G dimension in the loop ranges alongside a C or K, it means we have a 5D tensor,
            # where the onnx tensors are always flattened back to 4D (merging the G+C or G+K into one channel dimension)
            dimensions, loop_ranges = flatten_grouped_convolution_ranges(dimensions, loop_ranges)
            bounding_box = [loop_ranges[dim] for dim in dimensions]


            if not interleaved:
                bounding_box_flat = tuple([item for sublist in bounding_box for item in sublist])
                return bounding_box_flat
            else:
                bounding_box_flat = tuple(zip(*bounding_box))
                bounding_box_flat = tuple([item for sublist in bounding_box_flat for item in sublist])
                return bounding_box_flat
        
        def bounding_box_generator(nodes, operand):
            """
            Generator function that yields the bounding boxes of an operand for all nodes.
            """
            for i, node in enumerate(nodes):
                inclusive_ranges = convert_to_inclusive_data_range(node.loop_ranges)
                dimensions = node.operand_dimensionality_order[operand]
                bounds = get_bounding_box_dimensions(dimensions, inclusive_ranges)
                yield (i, bounds, None)
        
        def get_nb_input_dimensions(node):
            """Return the number of input dimensions this node has.
            We take the first non-constant input operand.

            Args:
                node (_type_): A Node object.
            """
            input_operand = list(set(node.input_operands) - set(node.constant_operands))[0]
            dims = node.operand_dimensionality_order[input_operand]
            if 'G' in dims and ('C' in dims or 'K' in dims):
                nb_dims = len(dims) - 1  # because later the generator will merge them into a single channel dim
            else:
                nb_dims = len(dims)
            return nb_dims

        def build_rtree(nodes, operand):
            """
            Build an rtree data structure based on each node in 'nodes' for the relevant dimensions of operand.
            """
            props = index.Property()
            props.dimension = get_nb_input_dimensions(nodes[0])  # We assume all nodes in 'nodes' have identical dimensions

            rtree = index.Index(bounding_box_generator(nodes, operand), properties=props)
            return rtree

        # Check all the different input operands of the consumer node that stem from the producer node
        # The direct predecessor of an input operand might be a DummyNode so we need to propagate back
        dependent_input_operands = []
        for operand, parent_node in consumer.input_operand_source.items():
            if parent_node == producer:
                dependent_input_operands.append(operand)
            elif parent_node:
                non_dummy_parents = self.get_non_type_predecessors(parent_node, [DummyNode])
                if producer in non_dummy_parents:
                    dependent_input_operands.append(operand)

        # edges will hold the cns that are dependent on each other [(prod_cn, cons_cn), ...]
        edges = []

        for input_operand in dependent_input_operands:
            # Assert that the input operand of this consumer and the output opernad of the producer have the same number of dimensions
            producer_dimensions = producer.operand_dimensionality_order['O']
            consumer_dimensions = consumer.operand_dimensionality_order[input_operand]
            # assert len(producer_dimensions) == len(consumer_dimensions), f"The producer {producer} has dimensions {producer_dimensions}, while the consumer {consumer} has dimensions {consumer_dimensions}."
            # Build the tree of all finer consumer nodes for this operand
            consumer_tree = build_rtree(finer_consumers, input_operand)

            # As long as we haven't iterated through all of the output's operand's irrelevant dimensions,
            # we shouldn't add an edge to the consumer layer's nodes, as this would create unnecessary graph complexity
            # Because we have the intra-edges between the nodes, and because the nodes irrelevant loops are incrementing,
            # we can make the graph simpler by just having one edge at the final irrelevant loop iteration producer node.
            # Get the relevant (including partially relevant) and irrelevant dimensions of the producer node's output
            producer_r_dims_output = producer.operand_dimensionality_order['O']
            producer_ir_dims_output = producer.operand_loop_dim['O']['ir']

            # Iterate through all the producer nodes and get the consumer nodes that require its outputs,
            # taking into account that we only want an edge if the producer's irrelevant loops are at a max
            for finer_producer in finer_producers:
                # Get the output irrelevant loop ranges and check if they are at least at the max
                ir_dims_not_at_max = [finer_producer.loop_ranges[ir_dim][1] < producer.loop_ranges[ir_dim][1] for ir_dim in producer_ir_dims_output]
                if any(ir_dims_not_at_max):
                    continue  # to the next finer producer

                p_inclusive_ranges = convert_to_inclusive_data_range(finer_producer.loop_ranges)
                p_bounding_box = get_bounding_box_dimensions(producer_r_dims_output, p_inclusive_ranges)

                # Get the finer consumer node ids that intersect with this finer producer node
                intersecting_consumer_node_ids = consumer_tree.intersection(p_bounding_box)

                for intersecting_consumer_node_id in intersecting_consumer_node_ids:
                    intersecting_consumer = finer_consumers[intersecting_consumer_node_id]
                    # Create a new communication node that will reside between the producer and consumer node
                    edges += [(finer_producer, intersecting_consumer, {'operand': input_operand, 'bits': finer_producer.data_produced_unique})]

                # edges += [(finer_producer, finer_consumers[consumer_node_id]) for consumer_node_id 
                #     in consumer_tree.intersection(p_bounding_box)]
        
        return edges

    def get_inter_edges_numpy(self, producer, consumer, finer_producers, finer_consumers):
        numpy_tensors = {}
        # Get the paths from producer to consumer
        paths_between_generator = nx.all_simple_paths(self.workload, source=producer, target=consumer)
        all_inter_edges = []
        for path_between in paths_between_generator:
            dependent_operand = 'O'
            # print(path_between)
            ## FIRST NODE
            # First node in the path is a ComputationNode,
            # of which we extract the output operand dependency tensor
            node = path_between[0]
            assert isinstance(node, ComputationNode), "First node in path should be ComputationNode"
            if node in numpy_tensors:
                tensor_cns = numpy_tensors[node]
            else:
                finer_nodes = self.finer_nodes_dict[node]
                tensor_cns = self.get_tensor_cns(node, finer_nodes)
                numpy_tensors[node] = tensor_cns
                tensor = tensor_cns['O']
            ## INTERMEDIATE NON-COMPUTATION NODES
            for i, node in enumerate(path_between[1:-1], start=1):
                if isinstance(node, ComputationNode):
                    raise ValueError("Intermediate nodes should not be of type ComputationNode.")
                tensor = self.propagate_cn_production_for_non_cn(node, tensor)
            ## LAST NODE IN PATH
            last_node = path_between[-1]
            # Find the operand for which this last node connects to its predecessor
            dependent_operand = next(op for op, dependent_node in last_node.input_operand_source.items() if dependent_node == node)
            if last_node in numpy_tensors:
                tensor_cns = numpy_tensors[last_node]
            else:
                finer_nodes = self.finer_nodes_dict[last_node]
                tensor_cns = self.get_tensor_cns(last_node, finer_nodes)
                numpy_tensors[node] = tensor_cns
            last_tensor = tensor_cns[dependent_operand]
            inter_edges = self.get_inter_edges_tensor_based(tensor, last_tensor)
            for (prod, cons) in inter_edges:
                all_inter_edges.append((prod, cons, {'operand': dependent_operand, 'bits': prod.data_produced_unique}))
        return all_inter_edges

    def propagate_cn_production_for_non_cn(self, node, input_tensor):
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
    def get_inter_edges_tensor_based(producer_output_tensor, consumer_input_tensor):
        """This method obtains the edges between a producer and consumer.
        This is done by iterating through all finer consumer nodes,
        for each consumer node we create a window and get all the producer nodes that produced this data window.

        Args:
            producer_output_tensor (np.ndarray): A tensor containing for each position which CNs will produce it
            consumer_input_tensor (np.ndarray): A tensor containing for each position which CNs will consume it
        """
        assert producer_output_tensor.shape == consumer_input_tensor.shape, "Arrays to construct inter-layer edges must be equal shape."
        inter_edges = set()
        for producer_set, consumer_set in zip(producer_output_tensor.flat, consumer_input_tensor.flat):
            if consumer_set is None:  # Happens for downsample layers (e.g. ComputationNode((16,)) for MBNetV2)
                continue
            for prod, cons in itertools.product(producer_set, consumer_set):
                inter_edges.add((prod, cons))
        # inter_edges_with_data = []
        # for (prod, cons) in inter_edges:
        #     bits_to_be_transferred = prod.data_produced_unique
        #     inter_edges_with_data.append((prod, cons, {'bits': bits_to_be_transferred}))
        # return inter_edges_with_data
        return inter_edges

    def get_tensor_cns(self, node, finer_nodes):
            is_source_node = len(self.get_non_type_predecessors(node, [DummyNode])) == 0
            variable_operands = node.variable_input_operands + [node.output_operand]
            tensor_dims = {op: node.operand_dimensionality_order[op] for op in variable_operands}
            all_loop_dim_sizes = node.loop_dim_size | node.pr_loop_dim_size  # union of dicts
            tensor_shapes = {op: tuple([all_loop_dim_sizes[dim] for dim in dims]) for (op, dims) in tensor_dims.items()}
            tensors_cns = {op: np.ndarray(shape, dtype=set) for (op, shape) in tensor_shapes.items()}  # Initial arrays
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
                    ir_dims_output = node.operand_loop_dim['O']['ir']
                    finer_nodes_list = finer_nodes  # list in regular order
                    should_add_to_tensor_list = [all([finer_node.loop_ranges[ir_dim][1] >= node.loop_ranges[ir_dim][1] for ir_dim in ir_dims_output]) for finer_node in finer_nodes_list]
                    attr_to_add_to = "data_produced_unique"
                    precision = node.operand_precision["O_final"]
                else:
                    finer_nodes_list = list(reversed(finer_nodes))  # list in reversed order
                    should_add_to_tensor_list = [True for finer_node in finer_nodes_list]
                    attr_to_add_to = "data_consumed_unique"
                    precision = node.operand_precision[op] * (not is_source_node)  # if this layer is the first layer, we assume the inputs are streamed and "free"
                nb_unique_data_seen = 0
                for finer_node, should_add_to_tensor in zip(finer_nodes_list, should_add_to_tensor_list):
                    if not should_add_to_tensor:
                        continue  # Skip if we're not at the max ir loop value for output
                    op_dim_ranges = [finer_node.loop_ranges[loop_dim] for loop_dim in dims]
                    op_dim_ranges_max_stop = tuple(tensor_shapes[op])
                    window = tuple([slice(max(0, start), stop) for (start, stop) in op_dim_ranges])  # start can be negative for padding which, makes np flip
                    # Count how many nans we have in this window, as this is the amount of unique data consumed/produced by this finer_node
                    nb_unique_data_bits = np.sum(tensors_cns[op][window] == set()) * precision
                    nb_unique_data_seen += nb_unique_data_bits
                    # Add this amount of unique data to the "data_consumed_unique" or "data_produced_unique" depending on input/output operand
                    setattr(finer_node, attr_to_add_to, getattr(finer_node, attr_to_add_to) + nb_unique_data_bits)
                    # Set this window of the tensor to indicate it will be consumed/produced by this finer node
                    bounded_op_dim_ranges = [range(max(0, start), min(max_stop, stop)) for ((start, stop), max_stop) in zip(op_dim_ranges, op_dim_ranges_max_stop)]
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
    def set_base_priority_of_nodes(G, finer_nodes_dict):
        """Set the base_priority of all stored tensors of variable operands in every node in finer_nodes
         based on the amount of real (excluding same layer edges) edges.

        Args:
            finer_nodes (list): List of the nodes for which to set the tensors' base_priority
        """
        nb_nodes_per_layer_id = {layer.id[0]: len(finer_nodes_dict[layer]) for layer in finer_nodes_dict.keys()}
        nb_seen_nodes_per_layer_id = {layer_id: 0 for layer_id in nb_nodes_per_layer_id.keys()}
        for node in nx.topological_sort(G):
            layer_id = node.id[0]
            for layer_operand in node.operand_list:
                tensor: Tensor = node.operand_tensors[layer_operand]
                if layer_operand == node.output_operand:
                    # Look at the amount of successors from different layers
                    successors = [succ for succ in G.successors(node) if succ.id[0] != layer_id]
                    tensor.set_base_priorities(len(successors))
            nb_seen_nodes_per_layer_id[layer_id] += 1

    def set_nb_real_predecessors(self, G):
        """Set nb_real_predecessors attribute for each node in G.
        A real predecessor is a predecessor coming from a different layer.

        Args:
            G (DiGraph): Graph containing the nodes and edges.
        """
        for n in G.nodes():
            nb_real_predecessors = len(list(pred for pred in G.predecessors(n) if pred.id[0] != n.id[0]))
            n.set_nb_real_predecessors(nb_real_predecessors)


def deduce_tensor_reuse_factors(original_node, outer_temporal_loops) -> dict[list[int]]:
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
    r_ir_LUT = original_node.operand_loop_dim
    constant_operands = original_node.constant_operands
    r_ir_loop = {}
    for constant_operand in constant_operands:
        r_ir_loop[constant_operand] = []
        for loop in outer_temporal_loops:
            if loop.dimension in r_ir_LUT[constant_operand]['ir']:
                r_ir_loop[constant_operand].append(('ir', loop.size))
            else:
                r_ir_loop[constant_operand].append(('r', loop.size))

    # total_reuse_factor is the upper bound of the reuse factor that current layer CNs can reach
    total_reuse_factor = {op: prod([reuse_factor for (loop_type, reuse_factor) in r_ir_loop[op] if loop_type == 'ir']) for op in r_ir_loop.keys()}

    # data_reuse_factor initialization
    data_reuse_factor = {}
    for op in r_ir_loop.keys():
        data_reuse_factor[op] = []
        if r_ir_loop[op][0][0] == 'ir':
            data_reuse_factor[op] = [i for i in range(total_reuse_factor[op], total_reuse_factor[op]-r_ir_loop[op][0][1], -1)]
            below_ir_size = [r_ir_loop[op][0][1]]
        else:
            data_reuse_factor[op] = [total_reuse_factor[op] for _ in range(total_reuse_factor[op], total_reuse_factor[op]-r_ir_loop[op][0][1], -1)]
            below_ir_size = [1]

    # return if there is only 1 outer-CN loop
    if len(r_ir_loop[op]) == 1:
        return data_reuse_factor

    for op in r_ir_loop.keys():
        # deduce the data_reuse_factor if there is more than 1 outer-CN loop
        for idx, (loop_type, loop_size) in enumerate(r_ir_loop[op][1:]):
            origin_data_reuse_factor_list = pickle_deepcopy(data_reuse_factor[op])
            if loop_type == 'ir':
                below_ir_size.append(below_ir_size[-1] * loop_size)
                for size in range(1, loop_size):
                    # reduce the data reuse factor if meet an ir loop
                    data_reuse_factor[op] += [i-size*below_ir_size[idx] for i in origin_data_reuse_factor_list]
            else:
                below_ir_size.append(below_ir_size[-1])
                for size in range(1, loop_size):
                    # maintain the data reuse factor if meet a r loop
                    data_reuse_factor[op] += origin_data_reuse_factor_list

    return data_reuse_factor