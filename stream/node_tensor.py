import math
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

ARRAY_T: TypeAlias = NDArray[Any]


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

    def _get_pointer(self):
        return self.__node_count

    def _increment_and_get_pointer(self):
        """Get the index pointer in the last dimension. which points to the next free spot to allocate nodes.
        Automatically increments the pointer after each use. If the index exceeds the allocated space, an error is
        raised."""
        self.__node_count += 1
        pointer = self.__node_count
        if pointer >= self.__pre_allocation_size:
            raise IndexError
        return pointer

    @property
    def shape(self) -> None:  # type: ignore
        """Protect the original shape attribute to prevent errors"""
        raise ValueError("The numpy shape of NodeTensor is hidden in an abstraction layer. Call `tensor_shape` instead")

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
        extended_slices = slices + (slice(0, self._get_pointer() + 1),)
        tensor_slice = self.as_ndarray()[extended_slices]
        all_empty_mask = np.all(tensor_slice == 0, axis=-1)
        return int(np.count_nonzero(all_empty_mask))

    def extend_with_node(self, slices: tuple[slice, ...], node: object) -> "NodeTensor":
        assert self.is_valid_shape_dimension(slices), "Last dimension of tensor is reserved for CNs"

        # Case 1: Try to assign at the current pointer for given slices
        idx = self._get_pointer()
        extended_slices = slices + (slice(idx, idx + 1),)
        assert all(s.stop <= self.full_shape[i] for i, s in enumerate(extended_slices)), "Index out of bounds"

        # Slice is all 0
        if not np.any(self[extended_slices]):
            self[extended_slices] = node
            return self

        # Case 2: increment pointer and assign at empty slice
        try:
            idx = self._increment_and_get_pointer()
            extended_slices = slices + (slice(idx, idx + 1),)
            self[extended_slices] = node
            return self
        # Case 3: pointer exceeds the tensor's accumulation dimension -> increase tensor size
        except IndexError:
            # Happens when all allocated space has been used up. Create new one and double allocated space
            new_pre_alloc_size = 2 * self.__pre_allocation_size
            new_full_shape = self.full_shape[:-1] + (new_pre_alloc_size,)
            new_tensor_np = np.zeros(new_full_shape, dtype=object)
            new_tensor_np[..., : self.full_shape[-1]] = self
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

    def split(self, split_indices: list[int], axis: int) -> "list[NodeTensor]":
        axis = axis - 1 if axis < 0 else axis
        return [t.view(NodeTensor) for t in np.split(self.as_ndarray(), split_indices, axis=axis)]

    def slice(self, starts: int, ends: int, axis: int, steps: int = 1) -> "NodeTensor":
        assert ends >= 0, "Negative indices not supported"
        axis = len(self.tensor_shape) - 1 if axis < 0 else axis

        slices = [slice(None)] * len(self.full_shape)
        slices[axis] = slice(starts, ends, steps)

        return self.as_ndarray()[tuple(slices)].view(NodeTensor)

    def concat_with_empty(self, shape: tuple[int, ...], axis: int, variable_input_first: bool):
        empty_shape = self.convert_to_full_shape(shape)
        empty_tensor = np.zeros(empty_shape, dtype=object)
        axis = axis - 1 if axis < 0 else axis
        if variable_input_first:
            return np.concat((empty_tensor, self.as_ndarray()), axis=axis).view(NodeTensor)
        else:
            return np.concat((self.as_ndarray(), empty_tensor), axis=axis).view(NodeTensor)

    def concat_with_empty_both_sides(
        self, output_shape: tuple[int, ...], axis: int, slice_idx: int, axis_exists_in_input: bool = False
    ):
        """Return a new tensor with shape `output_shape` that is all zero, except for the slice at `slice_idx` in
        the given axis. This slice is equal to this instance.
        In other words, a new dimension is created and the current instance is sandwiched between tensors of zeros in
        this dimension"""
        full_shape = self.convert_to_full_shape(output_shape)
        full_tensor = np.zeros(full_shape, dtype=object)

        if axis_exists_in_input:
            assert len(output_shape) == len(self.tensor_shape)
            assert self.tensor_shape[axis] == 1
            # Remove the size-1 dimension
            slice_to_assign = np.squeeze(self, axis)
        else:
            assert len(output_shape) == len(self.tensor_shape) + 1
            slice_to_assign = self

        slices = [slice(None)] * len(full_shape)
        slices[axis] = slice_idx  # type: ignore

        full_tensor[tuple(slices)] = slice_to_assign
        return full_tensor.view(NodeTensor)

    def expand_dims(self, axis: int):
        """Insert a size-1 dimension at axis"""
        axis = axis - 1 if axis < 0 else axis
        return np.expand_dims(self.as_ndarray(), axis).view(NodeTensor)

    def __repr__(self):
        return f"NodeTensor{self.tensor_shape}[depth={self.__node_count}]"

    def __reduce__(self):
        return self.as_ndarray().__reduce__()
