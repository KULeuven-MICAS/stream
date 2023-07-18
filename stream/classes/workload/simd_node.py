from stream.classes.workload.computation_node import ComputationNode


class SimdNode(ComputationNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "simd"
