import copy

import networkx as nx

from zigzag.classes.workload.layer_node import LayerNode
from stream.classes.workload.computation_node import ComputationNode
from typing import Dict, Any
from networkx import DiGraph

import logging
logger = logging.getLogger(__name__)


class DNNWorkload(DiGraph):

    def __init__(self, workload: Dict[Any, Dict], mapping: Dict[Any, Dict], accelerator, **attr):
        """
        Collect all the algorithmic workload information here.
        :param workload: user-defined workload file (py).

        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)

        layer_id_to_obj = {}  # Lookup dict for id to LayerNode object translation
        self.layer_node_list = []
        workload_saved = copy.deepcopy(workload)

        for i, (layer_id, layer) in enumerate(workload.items()):
            ''' Add layer-core allocation to the layer attribute'''
            if layer['operator_type'] in mapping:
                core_allocation = mapping[layer['operator_type']]["core_allocation"]
            else:
                try:
                    core_allocation = mapping["default"]["core_allocation"]
                except:
                    raise ValueError(f"There is no mapping provided for layer {layer['operator_type']}, nor a default one.")
            layer['core_allocation'] = core_allocation

            ''' Add spatial mapping to the layer attribute '''
            spatial_mapping = self.get_spatial_mappings(accelerator, core_allocation)
            layer["spatial_mapping"] = spatial_mapping

            ''' Add temporal ordering to the layer attribute '''
            # TODO allow user to define fixed temporal loop order

            '''For each item in the dict generate the LayerNode and add it to the dnn graph G'''
            layer_name = layer['operator_type'] + '_' + str(layer_id)
            layer_input_names = [l['operator_type'] + '_' + str(l_id) + '_output' for (l_id, l) in workload_saved.items()
                                 if l_id in self.cat_lists_from_all_values_of_a_dict(layer['operand_source'])]
            layer_output_names = [layer['operator_type'] + '_' + str(layer_id) + '_output']

            if not layer_input_names:
                layer_input_names = ['the_first_input']
            if not layer_output_names:
                layer_input_names = ['the_last_output']
            logger.info(f"Parsed layer node {layer_name} | INPUT {layer_input_names} | OUTPUT {layer_output_names}")
            ''' Assume always define the final layer in the end '''
            produces_final_output = not layer_output_names
            layer_node = ComputationNode((layer_id,), layer, layer_name, layer_input_names, layer_output_names, produces_final_output, add_missing_node_attrs=True)
            '''Save this layer_id and LayerNode pair in the layer_id_to_obj dict'''
            layer_id_to_obj[layer_id] = layer_node
            self.add_node(layer_node)
            self.layer_node_list.append(layer_node)
            '''Find all of its operand sources and add edges accordingly'''
            edges = []
            for (op, parent_list) in layer.get('operand_source', {}).items():
                for parent_id in parent_list:
                    parent_layer = layer_id_to_obj[parent_id]
                    edges.append((parent_layer, layer_node))
                    layer_node.input_operand_source[op] = parent_layer
            self.add_edges_from(edges)

    def topological_sort(self):
        return nx.topological_sort(self)

    def get_node_with_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        raise ValueError("DNNWorkload instance does not have a node with the requested id")

    @staticmethod
    def get_spatial_mappings(accelerator, core_allocation):
        # If there is only one possible core allocation, set the spatial mapping as the one(s) of that core
        if isinstance(core_allocation, int):
            core = accelerator.get_core(core_allocation)
            spatial_mappings = core.dataflows
        elif (isinstance(core_allocation, list) and len(core_allocation) == 1):
            core = accelerator.get_core(core_allocation[0])
            spatial_mappings = core.dataflows
        else:
            spatial_mappings = None
        return spatial_mappings

    @staticmethod
    def cat_lists_from_all_values_of_a_dict(dict_to_cat: Dict[Any, list]) -> list:
        li_ca = []
        for li in dict_to_cat.values():
            li_ca.extend(li)
        return li_ca
