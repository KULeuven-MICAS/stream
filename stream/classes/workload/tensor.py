class Tensor:
    """Class to represent a data tensor.
    TODO: Add from which layer this tensor originates and its dimension ranges
    """

    def __init__(self, size: int, origin=None, layer_operand: str = None, loop_dimensions: tuple = None, loop_ranges: tuple = None):
        """Initialize the Tensor instance.

        Args:
            size (int): the size of the tensor in bits
            origin (ComputationNode): The computation node that consumes/produces this tensor
            layer_operand (str, optional): The layer operand to which this tensor belongs
            loop_dimensions (tuple, optional): The loop dimensions for this tensor
            loop_ranges (tuple, optional): The loop range span for the different dimensions of this operand
        """
        self.size = size
        self.origin = origin
        self.layer_operand = layer_operand
        self.memory_operand = self.origin.memory_operand_links[layer_operand]
        self.loop_dimensions = loop_dimensions
        self.loop_ranges = loop_ranges
        self.base_priority = None  # Will be set when we know how many successors this node has (static)
        self.core_priorities = {}  # For each core, successors the tensor is still going to be used by (dynamic)
        self.id = self.origin.id + (layer_operand,)

    def __str__(self) -> str:
        return f"Tensor({self.origin}, {self.layer_operand})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.size

    def __hash__(self) -> int:
        return hash((self.origin, self.layer_operand))

    def __lt__(self, __o: object) -> bool:
        return isinstance(__o, Tensor) and self.size < __o.size

    # def __eq__(self, __o: object) -> bool:
    #     return isinstance(__o, Tensor) and \
    #         self.origin.id[0] == __o.origin.id[0] and \
    #         self.layer_operand == __o.layer_operand and \
    #         self.loop_ranges == __o.loop_ranges

    def equality_hash(self):
        return hash((self.origin.id[0], self.layer_operand, self.loop_ranges))

    def set_base_priorities(self, base_priority):
        self.base_priority = base_priority

    def initialize_core_priorities(self, G, node):
        layer_id = node.id[0]
        successors = [succ for succ in G.successors(node) if succ.id[0] != layer_id]
        if self.layer_operand == node.output_operand:
            self.core_priorities = {successor.core_allocation: 0 for successor in successors}
            for successor in successors:
                self.core_priorities[successor.core_allocation] += 1
        else:
            self.core_priorities[self.origin.core_allocation] = self.base_priority

    def get_total_priority(self):
        return sum([priority for priority in self.core_priorities.values()])
