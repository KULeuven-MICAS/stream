from enum import unique
import itertools
from math import ceil, prod
from re import L
from typing import List, Dict
import networkx as nx
from networkx import DiGraph
import numpy as np
from rtree import index
from sympy import re
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.reshape_node import ReshapeNode

from zigzag.classes.mapping.temporal.temporal_loop import TemporalLoop
from stream.classes.workload.communication_node import CommunicationNode
from stream.classes.workload.computation_node import ComputationNode
from zigzag.classes.stages.Stage import Stage
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.workload.pooling_node import PoolingNode

import logging

from zigzag.utils import pickle_deepcopy
logger = logging.getLogger(__name__)

class GenerateCNWorkloadStage(Stage):
    """
    Class that transforms the layer-by-layer workload into finer CN workload graph.
    """
    def __init__(self, list_of_callables, *, workload, accelerator,
                 CN_define_mode, hint_loops, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.cores = accelerator.cores
        self.core_ids = [core.id for core in self.cores]
        self.node_output_tensors = {}  # dict that will store the producing cn id for each node's output tensor
        self.output_name_to_node = {}  # dict that will store for each output tensor name which node produced it

        # for CN node size case study, will be used in the function of get_outer_tmap_loop_dimensions
        self.CN_define_mode = CN_define_mode
        if CN_define_mode == 1:
            self.outer_CN_loops = hint_loops
        elif CN_define_mode == 2:
            self.inner_CN_loops = hint_loops
        elif CN_define_mode == 3:
            self.factor_loops = hint_loops
        else:
            raise ValueError(f"CN_define_mode can not be {self.CN_define_mode}.")

        # Save for each of the workload's nodes the finer nodes that will be generated
        self.finer_nodes_dict = {}

    def run(self):
        logger.info(f'Start GenerateCNWorkloadStage.')
        unique_finer_nodes = []
        G = nx.DiGraph()
        for node in nx.topological_sort(self.workload):
            if not isinstance(node, ComputationNode):  # If other node types shouldn't be included in finer node graph, add here
                self.node_output_tensors, self.output_name_to_node = self.update_node_output_tensors_for_non_cn(node, self.node_output_tensors, self.output_name_to_node)
                continue
            outer_temporal_loops = self.get_outer_tmap_loop_dimensions(node)
            finer_nodes, unique_nodes = self.get_finer_nodes(node, outer_temporal_loops)
            tensors_cns = self.get_tensor_cns(node, finer_nodes)
            self.finer_nodes_dict[node] = finer_nodes
            unique_finer_nodes += unique_nodes
            logger.info(f"Decoupled {node} to {len(finer_nodes)} smaller CNs with {len(unique_nodes)} unique ones.")
            # Compute the edges between nodes originating from one bigger node (intra-edges)
            intra_edges = self.get_intra_edges(finer_nodes)
            G.add_edges_from(intra_edges)

            # Generate the inter-layer edges. This code block is pretty messy, could probably be done cleaner
            for input_name, input_operand in zip(node.input_names, node.variable_input_operands):  # assume these two are aligned
                try:
                    pred_output_tensor = self.node_output_tensors[input_name]
                except KeyError:  # This means the input_name is an absolute input of the model
                    continue
                node_input_tensor = tensors_cns[input_operand]
                inter_edges = self.get_inter_edges_tensor_based(pred_output_tensor, node_input_tensor)
                G.add_edges_from(inter_edges, operand=input_operand)

            # Save this node's output tensor_cns to node_output_tensors
            node_output_tensor_name = node.output_names[0]
            self.node_output_tensors[node_output_tensor_name] = tensors_cns[node.output_operand]
            self.output_name_to_node[node_output_tensor_name] = node

        # Set the base_priority value of all nodes in G
        self.set_base_priority_of_nodes(G, self.finer_nodes_dict)

        # Set nb of real predecessors of all nodes in G
        self.set_nb_real_predecessors(G)

        logger.info(f"Finer graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        kwargs = self.kwargs.copy()
        kwargs["workload"] = G
        kwargs["accelerator"] = self.accelerator

        logger.info(f'Finished GenerateCNWorkloadStage.')
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

        yield None, None


        # for layer in nx.topological_sort(self.workload):
        #     if type(layer) == DummyNode:
        #         continue  # skip the DummyNodes
        #     kwargs = self.kwargs.copy()
        #     kwargs['layer'] = layer
        #     sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        #     for cme, extra_info in sub_stage.run():
        #         yield cme, (layer, extra_info)

    @staticmethod
    def set_base_priority_of_nodes(G, finer_nodes_dict):
        """Set the base_priority of all stored tensors in every node in finer_nodes based on the amount of real (excluding same layer edges) edges.

        Args:
            finer_nodes (list): List of the nodes for which to set the tensors' base_priority
        """
        nb_nodes_per_layer_id = {layer.id[0]: len(finer_nodes_dict[layer]) for layer in finer_nodes_dict.keys()}
        nb_seen_nodes_per_layer_id = {layer_id: 0 for layer_id in nb_nodes_per_layer_id.keys()}
        for node in nx.topological_sort(G):
            layer_id = node.id[0]
            for layer_operand in node.operand_list:
                tensor = node.operand_tensors[layer_operand]
                if layer_operand in node.constant_operands:
                    tensor.set_base_priority(nb_nodes_per_layer_id[layer_id] - nb_seen_nodes_per_layer_id[layer_id])
                elif layer_operand == node.output_operand:
                    # Look at the amount of successors from different layers
                    tensor.set_base_priority(len([succ for succ in G.successors(node) if succ.id[0] != layer_id]))
                else:
                    tensor.set_base_priority(1)  # This currently isn't used (Input tensors are not used)
            nb_seen_nodes_per_layer_id[layer_id] += 1

    @staticmethod
    def update_node_output_tensors_for_non_cn(node, node_output_tensors, output_name_to_node):
        """Update the node_output_tensors dict for non ComputationNode objects
        This is needed because possible reshapes of the node_output_tensors

        Args:
            node_output_tensors (_type_): _description_
            output_name_to_node (_type_): _description_

        Returns:
            _type_: _description_
            _type_: _description_
        """
        if isinstance(node, ReshapeNode):
            node_input_names = node.input_names
            assert len(node_input_names) == 1, "ReshapeNode has more than one input."
            node_input_name = node.input_names[0]
            node_output_names = node.output_names
            assert len(node_output_names) == 1, "ReshapeNode has more than one output."
            node_output_name = node_output_names[0]
            # Update the dict by adding an entry for the output based on the reshaped input
            input_tensor = node_output_tensors[node_input_name]
            output_tensor = node.reshape_operand_tensor(input_tensor)
            node_output_tensors[node_output_name] = output_tensor
            output_name_to_node[node_output_name] = node

        elif isinstance(node, FlattenNode):
            node_input_names = node.input_names
            assert len(node_input_names) == 1, "FlattenNode has more than one input."
            node_input_name = node_input_names[0]
            node_output_names = node.output_names
            assert len(node_output_names) == 1, "FlattenNode has more than one output."
            node_output_name = node_output_names[0]
            # Update the dict by adding an entry for the output based on the flattened input
            input_tensor = node_output_tensors[node_input_name]
            output_tensor = node.flatten(input_tensor)
            node_output_tensors[node_output_name] = output_tensor
            output_name_to_node[node_output_name] = node

        elif isinstance(node, ElementwiseNode):
            # Get the tensors for the two inputs
            node_input_names = node.input_names
            assert len(node_input_names) == 2, "ElementwiseNode does not have two inputs."
            node_output_names = node.output_names
            assert len(node_output_names) == 1, "ElementwiseNode does not have one output."
            node_output_name = node_output_names[0]
            # Update the dict by adding an entry for the output based on the joined inputs
            tensors = [node_output_tensors[input_name] for input_name in node_input_names]
            output_tensor = node.join(tensors[0], tensors[1])
            node_output_tensors[node_output_name] = output_tensor
            output_name_to_node[node_output_name] = node

        elif isinstance(node, DummyNode):
            # if len(node.input_names) != 1 or len(node.output_names) != 1:
            #     logger.warning(f"{node} has more than one input/output:  inputs= {node.input_names} outputs= {node.output_names}.")
            # Copy over the tensor from first input to first output
            node_input_name = node.input_names[0]
            node_output_name = node.output_names[0]
            input_tensor = node_output_tensors[node_input_name]
            output_tensor = input_tensor
            node_output_tensors[node_output_name] = output_tensor
            output_name_to_node[node_output_name] = node
        return node_output_tensors, output_name_to_node

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

        if self.CN_define_mode == 1:
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

        elif self.CN_define_mode == 2:
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

    def get_type_predecessors(self, node, type):
        """Find all self.workload nodes that are of instance 'type'.

        Args:
            node (Node): the node for which we intend to find all preds that are not of a type in types
            type (type): type of node we accept as a valid predecessor.
        """
        preds = list(self.workload.predecessors(node))
        while any([not isinstance(pred, type) for pred in preds]):
            # Find first node in list that is of any type in types
            skip_node = next(pred for pred in preds if not isinstance(pred, type))
            # Find its index
            idx = preds.index(skip_node)
            # Find its predecessors
            skip_node_preds = list(self.workload.predecessors(skip_node))
            # Pop the skip_node from the list of preds and append its preds to the list
            preds.pop(idx)
            preds += skip_node_preds
        return preds

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
                finer_nodes_list = finer_nodes  # list in regular order
                attr_to_add_to = "data_produced_unique"
                precision = node.operand_precision["O_final"]
            else:
                finer_nodes_list = reversed(finer_nodes)  # list in reversed order
                attr_to_add_to = "data_consumed_unique"
                precision = node.operand_precision[op] * (not is_source_node)  # if this layer is the first layer, we assume the inputs are streamed and "free"
            nb_unique_data_seen = 0
            for finer_node in finer_nodes_list:
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
    def get_finer_nodes(original_node, outer_temporal_loops) -> List[ComputationNode]:
        
        # Extract the original node id. This should be a tuple of length one.
        # The finer nodes we generate will have a tuple of length two, of format (original_node_id, finer_node_id)
        original_node_id = original_node.id[0]

        # Take away the outer_temporal_loops to create finer CNs for this node
        finer_node_attrs = original_node.attrs.copy()
        node_type = original_node.type
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

            # If all the output irrelevant loops are at a max, this is producing a final output, so set a flag
            original_node_output_ir_dims = original_node.operand_loop_dim['O']['ir']
            if all([dim_min_max[dim][1] >= original_node.loop_dim_size[dim] for dim in original_node_output_ir_dims]):
                produces_final_output = True
            else:
                produces_final_output = False



            # Create the computation node object with the computed ranges of the loop dimensions and number of dying pixels
            finer_node_id = (original_node_id, n)
            finer_node_name = f"ComputationNode({(original_node_id, n)})"
            finer_node_input_names = original_node.input_names
            finer_node_output_names = original_node.output_names
            finer_node = ComputationNode(finer_node_id, finer_node_attrs_copy, finer_node_name, finer_node_input_names, finer_node_output_names, produces_final_output)
            finer_node.type = node_type
            finer_nodes.append(finer_node)


        # Just take the first node as they are all equal for now.
        # TODO If some are different, this should be done more smartly
        unique_finer_nodes = [finer_nodes[0]]

        return finer_nodes, unique_finer_nodes


    @staticmethod
    def get_intra_edges(nodes):
        return [(nodes[node_id], nodes[node_id + 1], {'size': 0, 'bits': 0}) for node_id in range(len(nodes) - 1)]

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
        inter_edges_with_data = []
        for (prod, cons) in inter_edges:
            bits_to_be_transferred = prod.data_produced_unique
            inter_edges_with_data.append((prod, cons, {'bits': bits_to_be_transferred}))
        return inter_edges_with_data

    @staticmethod
    def get_key(dict, value):
        """Return the first key in the dict that has a value of 'value'.

        Args:
            dict (dict): The dictionary
            value (Any): The value
        """
        for key, val in dict.items():
            if val == value:
                return key
        raise ValueError(f"Value {value} is not in dict {dict}.")

    def set_nb_real_predecessors(self, G):
        """Set nb_real_predecessors attribute for each node in G.
        A real predecessor is a predecessor coming from a different layer.

        Args:
            G (DiGraph): Graph containing the nodes and edges.
        """
        for n in G.nodes():
            nb_real_predecessors = len(list(pred for pred in G.predecessors(n) if pred.id[0] != n.id[0]))
            n.set_nb_real_predecessors(nb_real_predecessors)