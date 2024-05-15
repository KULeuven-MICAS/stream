from stream.classes.workload.dnn_workload import DNNWorkloadStream
from zigzag.parser.workload_factory import LayerNodeFactory, WorkloadFactory as ZigZagWorkloadFactory
from zigzag.workload.layer_node import LayerNode


class WorkloadFactoryStream(ZigZagWorkloadFactory):
    """Generates a `Workload` instance from the validated and normalized user-provided data.
    Almost identical to ZigZagWorkloadFactory, apart from the return type: DNNWorkloadStream instead of
    DNNWorkload
    """

    def create(self) -> DNNWorkloadStream:
        node_list: list[LayerNode] = []

        for layer_data in self.workload_data:
            layer_node_factory = LayerNodeFactory(layer_data, self.mapping_data)
            layer_node = layer_node_factory.create()
            node_list.append(layer_node)

        return DNNWorkloadStream(node_list)
