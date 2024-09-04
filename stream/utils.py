import math
import os
import pickle
import pprint
from typing import TYPE_CHECKING, Any, Generic, Iterator, Literal, Sequence, TypeAlias, TypeVar, overload

import networkx as nx
import numpy as np
from networkx import DiGraph
from numpy.typing import NDArray
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.onnx_workload import ComputationNodeWorkload

ARRAY_T: TypeAlias = NDArray[Any]


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


def save_scme(scme: "StreamCostModelEvaluation", path: str):
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


def save_core_allocation(
    workload: "ComputationNodeWorkload", path: str, type: str = "fixed", format: str = "py"
) -> dict:
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


class NodeTensor(np.ndarray[Any, Any]):
    """An instance of this class stores all ComputationNodes that are needed for a certain element within the loop-
    based workload.
    The last dimension of this tensor is ALWAYS reserved to accumulate nodes, whereas the first dimensions represent
    the workload size.
    """

    def __new__(cls, x: np.ndarray[Any, Any], pre_allocation_size: int):
        return x.view(cls)

    def __init__(self, x: np.ndarray[Any, Any], pre_allocation_size: int):
        self.__pre_allocation_size = pre_allocation_size
        self.__node_count = 0

    def as_ndarray(self) -> NDArray[Any]:
        """Typecast to superclass. This is necessary because numpy dispatches methods based on the instance's type"""
        return self.view(np.ndarray)  # type: ignore

    @staticmethod
    def initialize_empty(shape: tuple[int, ...], pre_allocation_size: int = 8):
        """Initialize a NodeTensor to store ComputationNodes. The tensor has shape `(shape, pre_allocation_size)`.
        ComputationNodes are accumulated in the last dimension and space is pre-allocated in memory for performance"""
        return NodeTensor(np.zeros(shape + (pre_allocation_size,), dtype=object), pre_allocation_size)

    def _get_and_increment_pointer(self):
        """Get the index pointer in the last dimension. which points to the next free spot to allocate nodes.
        Automatically increments the pointer after each use. If the index exceeds the allocated space, an error is
        raised."""
        pointer = self.__node_count
        if pointer >= self.__pre_allocation_size:
            raise IndexError
        self.__node_count += 1
        return pointer

    @property
    def shape(self) -> None:  # type: ignore
        """Protect the original shape attribute to prevent errors"""
        raise ValueError("The numpy shape of NodeTensor is hidden in an abstraction layer")

    @property
    def full_shape(self):
        return super().shape

    @property
    def tensor_shape(self):
        """The part of the shape that corresponds to the workload space"""
        return self.full_shape[:-1]

    @property
    def flat(self) -> NDArray[Any]:  # type: ignore
        """Iterates over all 'elements', i.e. iterates over the groups of nodes"""
        depth = math.prod(self.tensor_shape)
        new_full_shape = self.convert_to_full_shape((depth,))
        flattened = np.reshape(self.as_ndarray(), new_full_shape)
        return flattened

    def is_valid_shape_dimension(self, shape: tuple[int | slice, ...]):
        return len(shape) == len(self.tensor_shape)

    def convert_to_full_shape(self, tensor_shape: tuple[int, ...]):
        """Convert the given shape (where each dimension represents a workload dimension), to a full shape, with an
        added dimension to accumulate nodes."""
        return tensor_shape + (self.full_shape[-1],)

    def get_nb_empty_elements(self, slices: tuple[slice, ...]):
        """Returns the number of points for which there are no ComputationNodes."""
        assert self.is_valid_shape_dimension(slices), "Last dimension of tensor is reserved for CNs"
        extended_slices = slices + (slice(0, self.__node_count),)
        tensor_slice = self.as_ndarray()[extended_slices]
        all_empty_mask = np.logical_and.reduce(tensor_slice == 0, axis=-1)
        return int(np.sum(all_empty_mask))

    def extend_with_node(self, slices: tuple[slice, ...], node: object) -> "NodeTensor":
        assert self.is_valid_shape_dimension(slices), "Last dimension of tensor is reserved for CNs"

        try:
            idx = self._get_and_increment_pointer()
            extended_slices = slices + (slice(idx, idx + 1),)
            self[extended_slices] = node
            return self
        except IndexError:
            # Happens when all allocated space has been used up. Create new one and double allocated space
            new_tensor_np = np.concat((self, np.zeros(self.full_shape, dtype=object)), axis=-1)
            new_tensor = NodeTensor(new_tensor_np, pre_allocation_size=2 * self.__pre_allocation_size)
            # Update the node pointer
            new_tensor.__node_count = self.__node_count
            new_tensor = new_tensor.extend_with_node(slices, node)
            return new_tensor

    def reshape(self, new_shape: tuple[int, ...] | None) -> "NodeTensor":  # type: ignore
        """Wrap the numpy reshape method such that the user is agnostic to the last dimension on which nodes are
        accumulated"""
        if not new_shape:
            return self
        new_tensor_shape = self.convert_to_full_shape(new_shape)
        return np.reshape(self.as_ndarray(), new_tensor_shape).view(NodeTensor)

    def transpose(self, axes: list[int] | None):
        if axes is not None:
            # If `axes` contains `-1`, this represents the last workload dimension. We must incorporate the fact that
            # for `NodeTensor`, there is an extra, hidden dimension at the end
            axes = [(i - 1 if i < 0 else i) for i in axes]
            # Leave last dimension unchanged
            axes = axes + [len(self.tensor_shape)]

        return (np.transpose(self.as_ndarray(), axes=axes)).view(NodeTensor)

    def gather(self, gather_indices: int | list[int], axis: int) -> "NodeTensor":
        axis = axis - 1 if axis < 0 else axis
        return (np.take(self.as_ndarray(), gather_indices, axis=axis)).view(NodeTensor)

    def concat_with_empty(self, shape: tuple[int, ...], axis: int, variable_input_first: bool):
        emtpy_shape = self.convert_to_full_shape(shape)
        empty_tensor = np.zeros(emtpy_shape, dtype=object)
        axis = axis - 1 if axis < 0 else axis
        if variable_input_first:
            return np.concat((empty_tensor, self.as_ndarray()), axis=axis).view(NodeTensor)
        else:
            return np.concat((self.as_ndarray(), empty_tensor), axis=axis).view(NodeTensor)

    def __repr__(self):
        return f"TensorNode{self.tensor_shape}[depth={self.__node_count}]"

    def __reduce__(self):
        return self.as_ndarray().__reduce__()


T = TypeVar("T")


class DiGraphWrapper(Generic[T], DiGraph):
    """Wraps the DiGraph class with type annotations for the nodes"""

    @overload
    def in_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]:
        ...  # type: ignore

    @overload
    def in_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]:
        ...  # type: ignore

    @overload
    def in_edges(self, node: T) -> list[tuple[T, T]]:
        ...  # type: ignore

    def in_edges(  # type: ignore
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().in_edges(node, data)  # type: ignore

    @overload
    def out_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]:
        ...  # type: ignore

    @overload
    def out_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]:
        ...  # type: ignore

    @overload
    def out_edges(self, node: T) -> list[tuple[T, T]]:
        ...  # type: ignore

    def out_edges(  # type: ignore
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().out_edges(node, data)  # type: ignore

    def in_degree(self) -> Iterator[tuple[T, int]]:  # type: ignore
        return super().in_degree()  # type:ignore

    def out_degree(self) -> Iterator[tuple[T, int]]:  # type: ignore
        return super().out_degree()  # type:ignore

    def successors(self, node: T) -> Iterator[T]:  # type: ignore
        return super().successors(node)  # type: ignore

    def predecessors(self, node: T) -> Iterator[T]:  # type: ignore
        return super().predecessors(node)  # type: ignore

    def topological_sort(self) -> Iterator[T]:
        return nx.topological_sort(self)  # type: ignore

    def add_node(self, node: T) -> None:  # type: ignore
        super().add_node(node)  # type: ignore

    def add_nodes_from(self, node: Sequence[T]) -> None:  # type: ignore
        super().add_nodes_from(node)  # type: ignore

    def remove_nodes_from(self, nodes: Iterator[T]) -> None:
        super().remove_nodes_from(nodes)  # type: ignore

    def add_edge(self, edge_from: T, edge_to: T) -> None:  # type: ignore
        super().add_edge(edge_from, edge_to)  # type: ignore

    def add_edges_from(self, edges: Sequence[tuple[T, T] | tuple[T, T, Any]]) -> None:  # type: ignore
        super().add_edges_from(edges)  # type: ignore

    def successors(self, node: T) -> Iterator[T]:  # type: ignore
        return super().successors(node)  # type: ignore

    def predecessors(self, node: T) -> Iterator[T]:  # type: ignore
        return super().predecessors(node)  # type: ignore

    def all_simple_paths(self, producer: T, consumer: T) -> Iterator[list[T]]:
        return nx.all_simple_paths(self, source=producer, target=consumer)  # type: ignore

    def shortest_path(self, producer: T, consumer: T) -> list[T]:
        return nx.shortest_path(self, producer, consumer)  # type: ignore

    @property
    def node_list(self) -> list[T]:
        return list(self.nodes())  # type: ignore

    def get_node_with_id(self, node_id: int) -> T:
        for node in self.node_list:
            if node.id == node_id:  # type: ignore
                return node
        raise ValueError(f"Node with id {node_id} not found.")
