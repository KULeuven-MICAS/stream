import logging
import os
import pickle
import pprint
from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving

from stream.cost_model.core_cost import CoreCostEntry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

ARRAY_T: TypeAlias = NDArray[Any]


def get_too_large_operands(
    cme: CoreCostEntry | CostModelEvaluation, accelerator: "Accelerator", core_id: int
) -> list[MemoryOperand]:
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id: The id of the core of which we wish to get the too large operands.
    """
    too_large_operands: list[MemoryOperand] = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    mapping = getattr(cme, "mapping", None)
    if not mapping or not hasattr(mapping, "data_elem_per_level"):
        return too_large_operands
    for layer_operand, lvl in mapping.data_elem_per_level.items():
        memory_operand = cme.layer.memory_operand_links.layer_to_mem_op(layer_operand)
        if len(lvl) > core_nb_memory_levels[memory_operand] + 1:  # +1 because of spatial level
            too_large_operands.append(memory_operand)
    return too_large_operands


def save_core_allocation(
    workload: "ComputationNodeWorkload", path: str, type: str = "fixed", format: str = "py"
) -> dict:
    """Saves the core allocations of a workload to a python or pickle file.
    In fixed mode: if a layer has been split into multiple groups, the allocation of each group is saved to a tuple.
    In flexible mode: for each layer, the possible allocations are saved to a list.
    # TODO: Update this function to work with new mapping definition

    Args:
        workload (DiGraph): The graph of CNs
        path (str): The filepath to save the dict to.
        type (str, optional): The type of core allocation: fixed or flexible.

    Returns:
        allocations: The dictionary containing core allocations for each node name
    """
    node_allocations = {}
    node_allocations_grouped = {}
    for n in workload.node_list:
        if n.name not in node_allocations:
            node_allocations[n.name] = {"core_allocation": [n.chosen_core_allocation]}
            node_allocations_grouped[n.name] = {n.group: n.chosen_core_allocation}
        else:
            node_allocations[n.name]["core_allocation"].append(n.chosen_core_allocation)
            if n.group not in node_allocations_grouped[n.name]:
                node_allocations_grouped[n.name][n.group] = n.chosen_core_allocation
    if type == "fixed":
        mapping = {
            k: {"core_allocation": tuple(list(zip(*sorted(v.items()), strict=False))[1])}
            for k, v in node_allocations_grouped.items()
        }
    else:
        mapping = {k: {"core_allocation": sorted(set(v["core_allocation"]))} for k, v in node_allocations.items()}
    # Create folder structure if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The dict is saved with variable name 'mapping' as this is expected for running
    if format in ["python", "py"]:
        assert path.split(".")[-1] == "py", "format is python but file path doesn't end in .py"
        with open(path, "w") as handle:
            handle.write("mapping = ")
            handle.write(pprint.pformat(mapping))
    elif format in ["pickle", "pkl"]:
        with open(path, "wb") as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Invalid format: {format}.")
    return mapping


def get_unique_nodes(workload: "ComputationNodeWorkload") -> list["ComputationNode"]:
    """! Get the unique nodes from a workload."""
    unique_nodes: list[ComputationNode] = []
    for node in workload.node_list:
        equal_nodes = list(unique_node for unique_node in unique_nodes if node.has_same_performance(unique_node))
        if not equal_nodes:
            unique_nodes.append(node)
    return unique_nodes


def contains_wildcard(tiling) -> bool:
    """Returns wether the given tiling contains a wildcard number `*`. The wildcard must later be replaced by the
    constraint optimization into the optimal number of tiles"""
    return any(tiling == "*" for _, tiling in tiling)


def return_tiling_type(tiling):
    if contains_wildcard(tiling):
        raise ValueError(
            "Tiling contains wildcard. Use `replace_wildcard_tiling` to replace the wildcard with a number of tiles."
        )
    return tiling  # type: ignore


def get_inter_core_tiling_size(node) -> int:
    inter_core_tiling = node.inter_core_tiling
    if inter_core_tiling and not contains_wildcard(inter_core_tiling):
        total_tiling_size = 1
        for tiling_dim, tiling_size in inter_core_tiling:
            if tiling_size == "all":
                # If the inter_core_tiling is 'all', we assume it means all cores in the layer
                # and return the size of the layer dimension.
                assert node.layer_dim_sizes, "Layer dimension sizes must be defined for 'all' inter_core_tiling."
                assert isinstance(node.layer_dim_sizes, dict), "Layer dimension sizes must be a dictionary."
                tiling_size_updated = node.layer_dim_sizes[tiling_dim]
            else:
                tiling_size_updated = tiling_size
            assert isinstance(tiling_size_updated, int), f"Tiling size must be an integer, got {tiling_size}."
            total_tiling_size *= tiling_size_updated
        return total_tiling_size
    return 1
