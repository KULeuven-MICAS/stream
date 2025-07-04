from stream.workload.steady_state.tensor import SteadyStateTensor


class SteadyStateRollingBuffer(SteadyStateTensor):
    """
    Symbolic representation of a rolling buffer that contains multiple logically
    consecutive versions of a single tensor across steady-state iterations.
    """

    def __init__(
        self,
        base_tensor: SteadyStateTensor,
        num_tensors: int,
    ):
        id = base_tensor.id
        node_name = f"{base_tensor.node_name}_RB{num_tensors}"
        operand = base_tensor.operand
        steady_state_iteration_space = base_tensor.steady_state_iteration_space
        possible_resource_allocation = base_tensor.possible_resource_allocation
        super().__init__(
            id=id,
            node_name=node_name,
            size=base_tensor.size * num_tensors,
            type=base_tensor.tensor_flag,
            operand=operand,
            steady_state_iteration_space=steady_state_iteration_space,
            possible_resource_allocation=possible_resource_allocation,
        )
        self._base_tensor = base_tensor
        self._num_tensors = num_tensors

    @property
    def base_tensor(self) -> SteadyStateTensor:
        return self._base_tensor

    @base_tensor.setter
    def base_tensor(self, new_tensor: SteadyStateTensor):
        self._base_tensor = new_tensor
        self.tensor_flag = new_tensor.tensor_flag
        self._update_size()

    @property
    def num_tensors(self) -> int:
        return self._num_tensors

    @num_tensors.setter
    def num_tensors(self, new_count: int):
        self._num_tensors = new_count
        self._update_size()

    def _update_size(self):
        """Recalculate total size based on current base tensor and count."""
        self.size = self._base_tensor.size * self._num_tensors

    def __str__(self):
        return f"RollingBufferTensor({self.node_name}, {self.num_tensors} × {self.base_tensor.size}B = {self.size}B)"

    def __repr__(self):
        return str(self)
