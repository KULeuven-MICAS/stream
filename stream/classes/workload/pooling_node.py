from stream.classes.workload.computation_node import ComputationNode


class PoolingNode(ComputationNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "pooling"
