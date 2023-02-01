from stream.classes.workload.node import Node


class VirtualNode(Node):
    """Class to represent a virtual root node for the scheduler.
    """
    def __init__(self, core_allocation):
        super().__init__(type="virtual", energy=0, runtime=0, core_allocation=core_allocation)