import logging
from typing import TypeAlias

from stream.workload.computation.computation_node import ComputationNode, LoopRanges

logger = logging.getLogger(__name__)

GroupAllocation: TypeAlias = dict[tuple[tuple[int, int], ...], int]


class GroupIdManager:
    def __init__(self, node: ComputationNode):
        self.__id_count = 0
        self.groups: GroupAllocation = {}
        self.node = node
        self.inter_core_tiled_dims = [layer_dim for layer_dim, _ in node.inter_core_tiling]

    def __get_and_raise_id(self):
        curr_id = self.__id_count
        self.__id_count += 1
        return curr_id

    def __extract_inter_core_ranges(self, tile_loop_ranges: LoopRanges):
        """Given the loop ranges of a tile, return a hashable identifier that can be used to determine wether this
        tile belongs on the same core as other tiles. Tiles with different ranges for the inter core tile dimensions
        belong on different cores"""
        return tuple([tile_loop_ranges[dim] for dim in self.inter_core_tiled_dims])

    def get_group_id(self, tile_loop_ranges: LoopRanges) -> int:
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

        if not self.node.constant_operands and len(self.node.core_allocation) == 1:
            # If the node can only be assigned to a single core, we give all nodes the same group id
            # This is to prevent the CostModelEvaluationLUT from identifying each node as unique
            # This is the case for e.g. 'Add' nodes if there is only a single 'Add' core
            return 0

        # Differentiate based on node's inter core tiling
        range_identifier = self.__extract_inter_core_ranges(tile_loop_ranges)

        # This tile belongs together with previously seen tiles
        if range_identifier in self.groups:
            return self.groups[range_identifier]

        # New group
        new_group_id = self.__get_and_raise_id()
        self.groups[range_identifier] = new_group_id
        return new_group_id
