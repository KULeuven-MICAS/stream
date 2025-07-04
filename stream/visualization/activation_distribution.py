import networkx as nx
import pandas as pd
import plotly.express as px

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.utils import prune_workload


def plot_activation_distribution(
    workload: ComputationNodeWorkload, order: list | None = None, fig_path: str = "outputs/distribution.html"
):
    """
    Plot the output tensor sizes throughout the network depth.
    The depth is determined through the topological generations sort of the workload.
    The output tensor size at depth d is defined as the alive tensors before processing
    the nodes in the d'th topological generation.
    """
    # Generate order of processing if not provided
    order = order or [node.id for gen in nx.topological_generations(workload) for node in gen]
    # Get the activation size per processed node
    df, max_size = get_sizes_per_node(workload, order=order)
    # Plot the sizes
    nb_iterations = max(df["Iteration"])
    print(f"Activation distribution: {nb_iterations} iterations; {len(df)} bars; {max_size:.3e} max")
    fig = px.bar(
        df,
        x="Layer",
        y="Size",
        animation_frame="Iteration",
        animation_group="Layer",
        hover_name="Size",
        range_y=[0, max_size],
    )
    total_animation_time = 5000  # in ms
    transition_to_frame_ratio = 0.2
    time_per_iteration = total_animation_time / nb_iterations
    frame_time = (1 - transition_to_frame_ratio) * time_per_iteration
    transition_time = transition_to_frame_ratio * time_per_iteration
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_time  # type: ignore
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = transition_time  # type: ignore
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.write_html(fig_path)
    pass


def get_sizes_per_node(workload, order):
    data = []
    workload = prune_workload(
        workload,
        keep_types=[
            ComputationNode,
        ],
    )
    node_ids = set(n.id for n in workload.nodes())
    order = [o for o in order if o in node_ids]
    layer_ids = [n.id[0] for n in workload.nodes()]
    source_layer_ids = [n.id[0] for n, in_degree in workload.in_degree() if in_degree == 0]
    source_nodes = [n for n in workload.nodes() if n.id[0] in source_layer_ids]
    alive_tensors = set((t for n in source_nodes for op, t in n.operand_tensors.items() if op in ["I", "A"]))
    alive_tensors = set()  # temp without first input
    nb_tensor_uses = {t: 1 for t in alive_tensors}
    nb_tensor_uses = {}  # temp without first input
    alive_sizes_per_node = []
    alive_names_per_node = []
    sizes_per_layer = []
    it = 0
    max_size = 0
    for node_id in order:
        node = next(n for n in workload.nodes() if n.id == node_id)
        if not isinstance(node, ComputationNode):
            continue
        alive_names_per_node.append(tuple(str(t) for t in alive_tensors))
        alive_sizes_per_node.append(sum(t.size for t in alive_tensors))
        sizes = [sum(t.size for t in alive_tensors if t.origin.id[0] == layer_id) for layer_id in layer_ids]
        max_size = max(max_size, *sizes)
        sizes_per_layer.append(sizes)
        data += [
            {"Iteration": it, "Layer": str(layer_id), "Size": size}
            for layer_id, size in zip(layer_ids, sizes, strict=False)
        ]
        # Tensors that will die by processing this node
        # Input tensors from source nodes
        # for t in alive_tensors.copy():
        #     if t.origin == node:
        #         nb_tensor_uses[t] -= 1
        #         if nb_tensor_uses[t] == 0:
        #             alive_tensors.remove(t)
        # Output tensors from previous nodes
        for pred in workload.predecessors(node):
            pred_output = next(t for op, t in pred.operand_tensors.items() if op == "O")
            nb_tensor_uses[pred_output] -= 1
            if nb_tensor_uses[pred_output] == 0:
                alive_tensors.remove(pred_output)
        # Tensors that will spawn by processing this node (assumes only one output tensor)
        output = next(t for op, t in node.operand_tensors.items() if op == "O")
        alive_tensors.add(output)
        nb_tensor_uses[output] = len([n for n in workload.successors(node) if n.id[0] != node.id])
        it += 1
    # Add state after processing last node
    alive_names_per_node.append(tuple(str(t) for t in alive_tensors))
    alive_sizes_per_node.append(sum(t.size for t in alive_tensors))
    sizes = [sum(t.size for t in alive_tensors if t.origin.id == layer_id) for layer_id in layer_ids]
    sizes_per_layer.append(sizes)
    max_size = max(max_size, *sizes)
    data += [
        {"Iteration": it, "Layer": str(layer_id), "Size": size}
        for layer_id, size in zip(layer_ids, sizes, strict=False)
    ]
    df = pd.DataFrame(data)
    return df, max_size
