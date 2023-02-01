from stream.classes.workload.node import Node


class CommunicationNode(Node):
    """Class that represents a communcation node which is inserted between two nodes but doesn't have any computational information.
    """
    def __init__(self, communication_core_id, input_names, output_names) -> None:
        """Initialize the communication node.
        This initializes the energy and runtime to 0 and identifies on which core id the communcation bus resides.

        Args:
            core_id (int): the core id on which the communication happens
        """
        super().__init__(type="communication", energy=0, runtime=0, core_allocation=communication_core_id, input_names=input_names, output_names=output_names)

