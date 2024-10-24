import os
import pickle
import pprint
from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray
from onnx import ModelProto, NodeProto
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator as Core
from zigzag.parser.onnx.utils import get_onnx_tensor_type

from stream.workload.mapping import TILING_T

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload

ARRAY_T: TypeAlias = NDArray[Any]


def get_onnx_input_shapes(node: NodeProto, onnx_model: ModelProto) -> tuple[list[int], list[int]]:
    if len(node.input) != 2:
        raise ValueError(f"Node {node.name} does not have two inputs")
    input_name1 = node.input[0]
    input_name2 = node.input[1]
    input_shape1 = get_onnx_tensor_type(input_name1, onnx_model).shape
    input_shape2 = get_onnx_tensor_type(input_name2, onnx_model).shape
    return input_shape1, input_shape2


def has_asymmetric_input_data(node: NodeProto, onnx_model: ModelProto):
    """Return true iff the node has two inputs and the input nodes have a different shape"""
    if len(node.input) != 2:
        return False

    input_shape1, input_shape2 = get_onnx_input_shapes(node, onnx_model)
    return input_shape1 != input_shape2


def get_too_large_operands(cme: CostModelEvaluation, accelerator: "Accelerator", core_id: int) -> list[MemoryOperand]:
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id (int): The id of the core of which we wish to get the too large operands.
    """
    too_large_operands: list[MemoryOperand] = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    for layer_operand, lvl in cme.mapping.data_elem_per_level.items():
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
        allocations (dict): The dictionary containing core allocations for each node name
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
            k: {"core_allocation": tuple(list(zip(*sorted(v.items())))[1])} for k, v in node_allocations_grouped.items()
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
        equal_nodes = list(
            (
                unique_node
                for unique_node in unique_nodes
                if node.has_same_performance(unique_node) and node.group == unique_node.group
            )
        )
        if not equal_nodes:
            unique_nodes.append(node)
    return unique_nodes


def contains_wildcard(tiling: TILING_T):
    """Returns wether the given tiling contains a wildcard number `*`. The wildcard must later be replaced by the
    constraint optimization into the optimal number of tiles"""
    return any(tiling == "*" for _, tiling in tiling)


class CostModelEvaluationLUT:
    """A class to store the cost model evaluations in a look-up table.
    The look-up table is a dictionary with the following structure:
    {
        node0: {
            core0: CostModelEvaluation,
            ...
        },
        ...
    }
    """

    def __init__(self, cache_path: str | None, load: bool = True):
        self.lut: dict["ComputationNode", dict[Core, CostModelEvaluation]] = {}
        self.cache_path = cache_path
        if load and self.cache_path and os.path.exists(self.cache_path):
            self.load()

    def load(self):
        if not self.cache_path:
            raise ValueError("No cache_path provided.")
        try:
            with open(self.cache_path, "rb") as fp:
                self.lut = pickle.load(fp)
        except Exception as e:
            raise ValueError(
                f"Could not load look-up table from {self.cache_path}. Try removing the file if it exists."
            ) from e

    def save(self):
        if not self.cache_path:
            raise ValueError("No cache_path provided.")
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.lut, fp)

    def add_cme(self, node: "ComputationNode", core: Core, cme: CostModelEvaluation, allow_overwrite: bool = True):
        """Add a CostModelEvaluation to the look-up table for a given node and core.
        If a node with equal performance already exists in the look-up table,
        the CostModelEvaluation is added to that node."""
        if not allow_overwrite and self.has_cme(node, core):
            raise ValueError(f"CostModelEvaluation for node {node} and core {core} already exists.")
        if node not in self.lut:
            self.lut[node] = {}
        self.lut[node][core] = cme

    def has_cme(self, node: "ComputationNode", core: Core):
        """Check if a CostModelEvaluation exists for a given node and core."""
        return self.get_equal_node(node) is not None and node in self.get_nodes() and core in self.lut[node]

    def get_cme(self, node: "ComputationNode", core: Core):
        """Retrieve the CostModelEvaluation for a given node and core."""
        if not self.has_cme(node, core):
            raise ValueError(f"No CostModelEvaluation found for node {node} and core {core}.")
        return self.lut[node][core]

    def get_equal_node(self, node: "ComputationNode"):
        """Retrieve the node in the look-up table that is equal to the given node."""
        if any((n.has_same_performance(node) for n in self.lut)):
            return next(n for n in self.lut if n.has_same_performance(node))
        else:
            return None

    def get_equal_core(self, node: "ComputationNode", core: Core):
        """Retrieve the core in the look-up table that is equal to the given core."""
        try:
            return next(c for c in self.lut[node] if c.has_same_performance(core))
        except StopIteration or KeyError:
            return None

    def get_nodes(self):
        return list(self.lut.keys())

    def get_cores(self, node: "ComputationNode"):
        return list(self.lut.get(node, {}).keys())

    def remove_cores_with_same_id(self, node: "ComputationNode", core: Core):
        """! Removes cores with the same id as core for node from the look-up table."""
        if node in self.lut:
            self.lut[node] = {c: v for c, v in self.lut[node].items() if c.id != core.id}
