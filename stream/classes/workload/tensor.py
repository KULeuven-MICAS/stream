class Tensor:
    """Class to represent a data tensor.
    TODO: Add from which layer this tensor originates and its dimension ranges
    """

    def __init__(
        self,
        size: int,
        origin=None,
        layer_operand: str = None,
        loop_dimensions: tuple = None,
        loop_ranges: tuple = None,
    ):
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
        self.base_priority = (
            None  # Will be set when we know how many successors this node has (static)
        )
        self.instance_priorities = {}
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

    def get_instance_priority(self, top_instance, memory_manager):
        if top_instance in self.instance_priorities:
            return self.instance_priorities[top_instance]
        else:
            # If the top_instance is not in the dict. it means the core_id is the core that generates the tensor.
            # We  then return as priority the sum of all priorities of top instances that are not sotring the tensor.
            storing_instances, _, _ = memory_manager.find_tensor(self)
            not_storing_instances = list(
                set(self.instance_priorities.keys()) - set(storing_instances)
            )
            not_storing_priority = sum(
                (
                    self.instance_priorities[not_storing_instance]
                    for not_storing_instance in not_storing_instances
                )
            )
            return not_storing_priority

    def initialize_instance_priorities(self, G, node, accelerator):
        if self.layer_operand == node.output_operand:
            out_edges = [
                (succ, d)
                for n, succ, d in G.out_edges(node, data=True)
                if succ.id[0] != n.id[0]
            ]
            for successor, data in out_edges:
                core = accelerator.get_core(successor.core_allocation)
                layer_operand = data["operand"]
                memory_operand = successor.memory_operand_links[layer_operand]
                top_instance = core.get_top_memory_instance(memory_operand)
                if top_instance in self.instance_priorities:
                    self.instance_priorities[top_instance] += 1
                else:  # first time we see this instance
                    self.instance_priorities[top_instance] = 1

        else:
            core = accelerator.get_core(node.core_allocation)
            memory_operand = self.memory_operand
            top_instance = core.get_top_memory_instance(memory_operand)
            self.instance_priorities[top_instance] = self.base_priority

    def get_total_priority(self):
        return sum(self.instance_priorities.values())
