import logging
from typing import Any, TypeAlias

from zigzag.datatypes import MemoryOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)

STACK_T: TypeAlias = tuple[int, ...]


class LayerStacksGenerationStage(Stage):
    layer_stacks: list[STACK_T] | None

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ComputationNodeWorkload,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload

        self.layer_stacks = kwargs.get("layer_stacks", None)
        self.mode = kwargs.get("mode")
        self.stack_cutoff = kwargs.get("stack_cutoff", None)
        self.stack_cutoffs = kwargs.get("stack_cutoffs", None)

        # Get the weight capacity of all cores
        weight_capacities: dict[int, int] = {}
        for core in self.accelerator.cores.node_list:
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            mem_op = MemoryOperand("I2")
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(mem_op).memory_instance.size
            weight_capacities[core.id] = core_weight_capacity

        # Total weight capacity in bits
        self.total_weight_capacity = sum(weight_capacities.values())

    def run(self):
        if self.mode == "fused":
            if self.layer_stacks is None:
                if self.stack_cutoff is not None:
                    self.layer_stacks = self.get_layer_stacks_fused_single_fixed()
                elif self.stack_cutoffs is not None:
                    self.layer_stacks = self.get_layer_stacks_fused_multiple_fixed()
                else:
                    self.layer_stacks = self.get_layer_stacks_fused_single()
            else:
                self.layer_stacks = self.fill_layer_stacks_to_completion()

        elif self.mode == "lbl":
            self.layer_stacks = self.get_layer_stacks_lbl()
        else:
            raise ValueError("Unsupported mode for layer stack determination.")

        self.only_keep_computation_node_ids()

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["layer_stacks"] = self.layer_stacks
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        yield from sub_stage.run()

    def only_keep_computation_node_ids(self):
        """! Update the layer stacks to only keep ids of ComputationNodes"""
        assert self.layer_stacks is not None
        updated_layer_stacks: list[tuple[int, ...]] = []
        for stack in self.layer_stacks:
            update_stack: list[int] = []
            for layer_id in stack:
                try:
                    # Ignore node ids that do not exist
                    n = next(n for n in self.workload.node_list if n.id == layer_id)
                    if isinstance(n, ComputationNode):
                        update_stack.append(layer_id)
                except StopIteration:
                    pass
            updated_layer_stacks.append(tuple(update_stack))
        self.layer_stacks = updated_layer_stacks

    def get_layer_stacks_lbl(self):
        return [(id,) for id in sorted([n.id for n in self.workload.node_list if isinstance(n, ComputationNode)])]

    def fill_layer_stacks_to_completion(self):
        assert self.layer_stacks is not None
        stacks: list[tuple[int, ...]] = self.layer_stacks

        for node in self.workload.node_list:
            if not any(node.id in stack for stack in stacks):
                stacks += [(node.id,)]
        return stacks

    def get_layer_stacks_fused(self):
        cumsum = 0
        stacks: list[tuple[int, ...]] = []
        current_stack: list[int] = []
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def get_layer_stacks_fused_single(self):
        """
        Only the first set of layers will be fused, rest layer by layer"""
        cumsum = 0
        stacks: list[tuple[int, ...]] = []
        current_stack: list[int] = []
        first_complete = False
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if first_complete:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    continue
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                    first_complete = True
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def get_layer_stacks_fused_single_fixed(self):
        """
        layers will be fused based on ids in stack cutoffs. if ratio of weights > 1, we switch to layer by layer
        """
        assert self.stack_cutoff is not None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if id > self.stack_cutoff:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def get_layer_stacks_fused_multiple_fixed(self):
        """
        Only the first set of layers will be fused until fixed id, rest layer by layer
        """
        assert self.stack_cutoffs is not None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        assert len(self.stack_cutoffs) > 0
        stack_cutoff = self.stack_cutoffs[0]
        cutoff_idx = 1
        cumsum = 0
        lbl = False  # flag to switch to layer by layer
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if lbl:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    continue
                if id > stack_cutoff:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = 0
                    if cutoff_idx <= len(self.stack_cutoffs) - 1:
                        stack_cutoff = self.stack_cutoffs[cutoff_idx]
                        cutoff_idx += 1
                    else:
                        lbl = True
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    if id not in current_stack:
                        current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                    lbl = True
                elif not lbl:
                    if id not in current_stack:
                        current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks
