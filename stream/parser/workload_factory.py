from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.parser.workload_factory import WorkloadFactory as ZigZagWorkloadFactory
from zigzag.workload.layer_node import LayerNode

from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import InterCoreMappingAttributes


class WorkloadFactoryStream(ZigZagWorkloadFactory):
    """Generates a `Workload` instance from the validated and normalized user-provided data.
    Almost identical to ZigZagWorkloadFactory, apart from the return type: DNNWorkloadStream instead of
    DNNWorkload
    """

    def create(self, all_mappings: dict[str, InterCoreMappingAttributes]) -> DNNWorkloadStream:  # type: ignore
        node_list: list[LayerNode] = []
        raise NotImplementedError("TODO")

        for layer_data in self.workload_data:
            # TODO: don't create layer note but only extract the attributes
            layer_node_factory = LayerNodeFactory(layer_data, self.mapping_data)
            layer_node = layer_node_factory.create()
            node_list.append(layer_node)

        return DNNWorkloadStream(node_list)
