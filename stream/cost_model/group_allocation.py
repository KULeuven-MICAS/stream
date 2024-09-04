import logging
from typing import TypeAlias

from stream.workload.computation_node import ComputationNode, LoopRanges

logger = logging.getLogger(__name__)

GroupAllocation: TypeAlias = dict[tuple[tuple[int, int], ...], int]


class GroupIdManager:
    def __init__(self):
        self.__id_count = 0
        self.groups: GroupAllocation = {}

    def __get_and_raise_id(self):
        curr_id = self.__id_count
        self.__id_count += 1
        return curr_id

    def get_group_id(self, node: ComputationNode, loop_ranges: LoopRanges) -> int:
        """Return the group id for the given loop ranges.
        The group id is determined based on the relevant constant operand dimension loop ranges.
        If there is no constant operand, we return 0.
        If there is more than one constant operand, we only consider the last one's loop ranges.
        If those loop ranges are already contained within 'groups' we return that group id.
        Else we add it to the groups dict with an incremented group id.

        Args:
            node (ComputationNode): The original (layer) CN.
            loop_ranges (dict): A dictionary containing the loop range for each dimension

        Returns:
            int: The group id for the given loop ranges
        """
        # No constant operand
        if not node.constant_operands:
            return self.__get_and_raise_id()

        # Constant operand and known ranges
        constant_operand = node.constant_operands[-1]
        relevant_dims = node.loop_relevancy_info.get_r_layer_dims(constant_operand)
        relevant_ranges = tuple([loop_ranges[dim] for dim in relevant_dims])
        if relevant_ranges in self.groups:
            return self.groups[relevant_ranges]

        # Constant operand and new ranges
        new_group_id = self.__get_and_raise_id()
        self.groups[relevant_ranges] = new_group_id
        return new_group_id
