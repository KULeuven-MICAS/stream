from typing import Any

from stream.classes.io.workload_factory import WorkloadFactoryStream
from stream.classes.workload.dnn_workload import DNNWorkloadStream
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.stages.WorkloadParserStage import WorkloadParserStage as ZigZagWorkloadParserStage
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.io.onnx.model import ONNXModelParser

import logging

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.mapping_path = mapping_path
        self.onnx_model_parser = ONNXModelParser(workload_path, mapping_path, accelerator)

    def run(self):
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["mapping_path"] = self.mapping_path
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


# def parse_workload_from_path(workload_path: str, mapping_path: str, accelerator: Accelerator):
#     """
#     Parse the input workload residing in workload_path.
#     """

#     # If workload_path is a string, then it is a path to a workload file
#     if isinstance(workload_path, str):
#         module = importlib.import_module(workload_path)
#         workload = module.workload

#     else:
#         raise NotImplementedError(f"Provided workload format ({type(workload_path)}) is not supported.")

#     module = importlib.import_module(mapping_path)
#     mapping = module.mapping

#     workload = DNNWorkload(workload, mapping, accelerator)
#     logger.info(
#         f"Created workload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges."
#     )

#     return workload


# class UserDefinedModelParserStage(Stage):
#     def __init__(
#         self,
#         list_of_callables: list[StageCallable],
#         *,
#         workload_path: str,
#         mapping_path: str,
#         accelerator: Accelerator,
#         **kwargs: Any,
#     ):
#         super().__init__(list_of_callables, **kwargs)
#         self.workload_path = workload_path
#         self.mapping_path = mapping_path
#         self.accelerator = accelerator

#     def run(self):
#         workload = parse_workload_from_path(self.workload_path, self.mapping_path, self.accelerator)

#         self.kwargs["accelerator"] = self.accelerator
#         sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
#         for cme, extra_info in sub_stage.run():
#             yield cme, extra_info


class UserDefinedModelParserStage(ZigZagWorkloadParserStage):
    """Parses a user-provided workload from a yaml file.
    This class is very similar to WorkloadParserStage from ZigZag, the main difference being that this class creates a
    (Stream)DNNWorkload of ComputationNodes, while the ZigZag variant creates a (ZigZag) DNNWorkload of LayerNodes
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables=list_of_callables, workload=workload_path, mapping=mapping_path, **kwargs)
        self.accelerator = accelerator

    def run(self):
        workload = self.parse_workload_stream()
        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload_stream(self) -> DNNWorkloadStream:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactoryStream(workload_data, mapping_data)
        return factory.create()

    # def __parse_workload_data(self) -> list[dict[str, Any]]:
    #     """Parse, validate and normalize workload"""
    #     workload_data = open_yaml(self.workload_yaml_path)
    #     workload_validator = WorkloadValidator(workload_data)
    #     workload_data = workload_validator.normalized_data
    #     workload_validate_succes = workload_validator.validate()
    #     if not workload_validate_succes:
    #         raise ValueError("Failed to validate user provided workload.")
    #     return workload_data
