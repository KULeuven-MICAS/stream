from enum import Enum

import networkx as nx
from zigzag.datatypes import Constants

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload


class LayerStackMode(Enum):
    STANDARD = 0
    OCCUPATION_BASED = 1
    MANUAL = 2


def fits(occupations: dict[int, int], core_capacities: dict[int, int], occupation_factor: float):
    return not any([occupations[core_id] >= core_capacities[core_id] * occupation_factor for core_id in occupations])


def update_occupations(
    workload: ComputationNodeWorkload, occupations: dict[int, int], layer_id: int, group_ids: list[int]
):
    for group_id in group_ids:
        # Find a node that has this layer and group id and extract its constant op size
        node = next(n for n in workload.node_list if n.id == layer_id and n.group == group_id)
        constant_operands = node.constant_operands
        if not constant_operands:
            continue
        # Assume last constant operand is correct one for stack calculation
        constant_operand = constant_operands[-1]
        # Assert that the memory operand matches the assumed one for capacities
        memory_operand = node.memory_operand_links[constant_operand]
        assert memory_operand == Constants.MEM_OP_2
        # Get the size of the constant operand and add it to the current stack
        size = node.operand_size_bit[constant_operand]
        allocation = node.chosen_core_allocation
        assert allocation is not None
        occupations[allocation] += size


def get_layer_stacks_standard(workload: ComputationNodeWorkload):
    """Return all layer ids in a single stack.

    Args:
        workload (DiGraph): The workload.
    """
    layer_ids = sorted(set(node.id for node in workload.node_list))
    return [layer_ids]


def get_layer_stacks_occupation_based(
    workload: ComputationNodeWorkload,  # cn-wise workload
    original_workload: ComputationNodeWorkload,  # layer-wise workload
    accelerator: Accelerator,
    occupation_factor: int,
):
    # Get all layer id, group combinations in the workload
    layer_groups: dict[int, set[int]] = {}
    for n in workload.node_list:
        if not isinstance(n, ComputationNode):
            continue
        layer_id = n.id
        group_id = n.group
        if layer_id in layer_groups:
            layer_groups[layer_id].add(group_id)
        else:
            layer_groups[layer_id] = {group_id}
    # Active cores given the allocations of the workload
    active_core_ids = sorted(
        set(n.chosen_core_allocation for n in workload.node_list if n.chosen_core_allocation is not None)
    )
    core_capacities: dict[int, int] = {
        core_id: accelerator.get_core(core_id).get_top_memory_instance(Constants.MEM_OP_2).size
        for core_id in active_core_ids
    }

    all_stacks: list[list[int]] = []
    current_stack: list[int] = []
    # Track the constant operand occupation in all cores for the current stack
    occupations = {core_id: 0 for core_id in active_core_ids}
    # Compute the layer cutoffs based on the topological generations
    # and the constant operands size
    for generation in nx.topological_generations(original_workload):
        for original_node in generation:
            if not isinstance(original_node, ComputationNode):
                continue
            layer_id = original_node.id
            group_ids = layer_groups[layer_id]
            update_occupations(workload, occupations, layer_id, sorted(group_ids))
            # Check if the occupation exceeds the capacity * occupation_factor for any core
            if fits(occupations, core_capacities, occupation_factor):
                # Add this layer to the current stack as it fits
                current_stack.append(layer_id)
            # If no layers in current stack, add the current layer and cut
            elif not current_stack:
                current_stack.append(layer_id)
                all_stacks.append(sorted(current_stack))
                # Reset the current stack and the current occupations
                current_stack = []
                # Reset the occupations
                occupations = {core_id: 0 for core_id in active_core_ids}
            # Else, cut the stack and add current layer as first one of next stack
            else:
                all_stacks.append(sorted(current_stack))
                # Reset the current stack to include the current layer that didn't fit
                current_stack = [layer_id]
                # Reset the occupations to include the current layer that didn't fit
                occupations = {core_id: 0 for core_id in active_core_ids}
                update_occupations(workload, occupations, layer_id, sorted(group_ids))

    if current_stack:
        all_stacks.append(current_stack)
    return all_stacks


def get_layer_stacks(
    workload: ComputationNodeWorkload,  # cn-wise workload
    original_workload: ComputationNodeWorkload,  # layer-wise workload
    accelerator: Accelerator,
    occupation_factor: int,
    mode: LayerStackMode,
    layer_stacks,
):
    match mode:
        case LayerStackMode.STANDARD:
            return get_layer_stacks_standard(workload)
        case LayerStackMode.OCCUPATION_BASED:
            return get_layer_stacks_occupation_based(
                workload,
                original_workload,
                accelerator,
                occupation_factor,
            )
        case LayerStackMode.MANUAL:
            assert layer_stacks
            return layer_stacks
