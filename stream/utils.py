import pickle
from networkx import DiGraph
import json
import os
import pprint

from stream.classes.cost_model.cost_model import StreamCostModelEvaluation


def get_too_large_operands(cme, accelerator, core_id):
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id (int): The id of the core of which we wish to get the too large operands.
    """
    too_large_operands = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    for layer_operand, l in cme.mapping.data_elem_per_level.items():
        memory_operand = cme.layer.memory_operand_links[layer_operand]
        if (
            len(l) > core_nb_memory_levels[memory_operand] + 1
        ):  # +1 because of spatial level
            too_large_operands.append(memory_operand)
    return too_large_operands


def save_scme(scme: StreamCostModelEvaluation, path: str):
    """Saves anSCME to a pickle file.

    Args:
        scme (StreamCostModelEvaluation): The stream cost model evaluation.
        path (str): The filepath to save the pickle to.
    """
    with open(path, "wb") as fp:
        pickle.dump(scme, fp)


def load_scme(path: str):
    """Loads an SCME from a pickle file path.

    Args:
        path (str): The pickle filepath
    """
    with open(path, "rb") as fp:
        scme = pickle.load(fp)
    return scme

def save_core_allocation(workload: DiGraph, path: str, type="fixed", format="py") -> dict:
    """Saves the core allocations of a workload to a python or pickle file.
    In fixed mode: if a layer has been split into multiple groups, the allocation of each group is saved to a tuple.
    In flexible mode: for each layer, the possible allocations are saved to a list.
    
    Args:
        workload (DiGraph): The graph of CNs
        path (str): The filepath to save the dict to.
        type (str, optional): The type of core allocation: fixed or flexible.
    
    Returns:
        allocations (dict): The dictionary containing core allocations for each node name
    """
    node_allocations = {}
    node_allocations_grouped = {}
    for n in workload.nodes(): 
        if n.name not in node_allocations:
            node_allocations[n.name] = {"core_allocation": [n.core_allocation]}
            node_allocations_grouped[n.name] = {n.group: n.core_allocation}
        else:
            node_allocations[n.name]["core_allocation"].append(n.core_allocation)
            if n.group not in node_allocations_grouped[n.name]:
                node_allocations_grouped[n.name][n.group] = n.core_allocation
    if type == "fixed":
        mapping = {k: {"core_allocation": tuple(list(zip(*sorted(v.items())))[1])} for k, v in node_allocations_grouped.items()}
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