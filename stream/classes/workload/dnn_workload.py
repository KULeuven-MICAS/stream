from typing import Any
from zigzag.workload.Workload import Workload
from zigzag.workload.layer_node import LayerNode
from stream.classes.workload.computation_node import ComputationNode

import logging

logger = logging.getLogger(__name__)


class DNNWorkloadStream(Workload):
    def __init__(self, nodes: list[LayerNode], **attr: Any):
        """
        Collect all the algorithmic workload information here.
        Similar to `DNNWorkload` from ZigZag, but returns a DiGraph of ComputationNodes instead of LayerNodes.

        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)

        layer_id_to_obj: dict[int, ComputationNode] = {}
        self.layer_node_list = nodes

        # workload_saved = copy.deepcopy(workload)

        for node in nodes:

            # Create ComputationNode
            node_name = f"{node.type}_{node.id}"
            node_input_names = [
                f"{other_layer_node.type}_{other_layer_node.id}"
                for other_layer_node in nodes
                if other_layer_node.id in node.input_operand_source.values()
            ]
            node_output_names = [f"{node_name}_output"]
            if len(node_input_names) == 0:
                node_input_names = ["the_first_input"]

            # Assume always define the final layer in the end
            # produces_final_output = not layer_output_names # TODO don't understand this old piece of code
            produces_final_output = False

            op_type = node.type.lower()
            node_attr = node.extract_node_attr()
            computation_node = ComputationNode(
                node_id=node.id,
                node_name=node_name,
                node_attr=node_attr,
                input_names=node_input_names,
                output_names=node_output_names,
                op_type=op_type,
                produces_final_output=produces_final_output,
            )

            # Add to graph
            logger.info("Parsed layer node %s | INPUT %s | OUTPUT %s", node_name, node_input_names, node_output_names)
            layer_id_to_obj[computation_node.id] = computation_node
            self.add_workload_node(computation_node)

            # Find all of its operand sources and add edges accordingly
            edges: list[tuple[LayerNode, LayerNode]] = []
            for _, parent_id in node.input_operand_source.items():
                # for parent_id in parent_list:
                if parent_id not in layer_id_to_obj:
                    raise ValueError(f"Illegal reference to non-existent layer with id {parent_id}")
                parent_node = layer_id_to_obj[parent_id]
                edges.append((parent_node, computation_node))
            self.add_workload_edges_from(edges)
