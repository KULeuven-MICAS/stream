import os
from math import ceil
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.shape_inference import infer_shapes

from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.dummy_node import DummyNode
from zigzag.stages.Stage import Stage

import logging

logger = logging.getLogger(__name__)


class DetermineLayerStacksStage(Stage):
    def __init__(self, list_of_callables, *, accelerator, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload

        self.layer_stacks = kwargs.get("layer_stacks", None)
        self.mode = kwargs.get("mode")
        self.stack_cutoff = kwargs.get("stack_cutoff", None)
        self.stack_cutoffs = kwargs.get("stack_cutoffs", None)

        # Get the weight capacity of all cores
        weight_capacities = {}
        for core in self.accelerator.cores.nodes():
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(
                "I2"
            ).memory_instance.size
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
        elif self.mode == "lbl":
            if self.layer_stacks is None:
                self.layer_stacks = self.get_layer_stacks_lbl()
        else:
            raise ValueError("Unsupported mode for layer stack determination.")

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["layer_stacks"] = self.layer_stacks
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def get_layer_stacks_lbl(self):
        return [
            (id,)
            for id in sorted(
                [
                    n.id[0]
                    for n in self.workload.nodes()
                    if isinstance(n, ComputationNode)
                ]
            )
        ]

    def get_layer_stacks_fused(self):
        cumsum = 0
        stacks = []
        current_stack = []
        for n in sorted(list(self.workload.nodes()), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id[0]
                try:
                    op = next(op for op in n.constant_operands)
                except:
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
        stacks = []
        current_stack = []
        first_complete = False
        for n in sorted(list(self.workload.nodes()), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id[0]
                if first_complete:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    continue
                try:
                    op = next(op for op in n.constant_operands)
                except:
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
        assert not self.stack_cutoff is None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        for n in sorted(list(self.workload.nodes()), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id[0]
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
        assert not self.stack_cutoffs is None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        assert len(self.stack_cutoffs) > 0
        stack_cutoff = self.stack_cutoffs[0]
        cutoff_idx = 1
        cumsum = 0
        lbl = False  # flag to switch to layer by layer
        for n in sorted(list(self.workload.nodes()), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id[0]
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
                except:
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
