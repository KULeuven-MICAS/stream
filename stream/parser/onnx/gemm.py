import logging
from typing import Generator

from zigzag.parser.onnx.gemm_parser import GemmParser as GemmParserZigZag

from stream.hardware.architecture.accelerator import Accelerator
from stream.classes.io.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation_node import ComputationNode

logger = logging.getLogger(__name__)


class GemmParser(GemmParserZigZag, OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def run(self) -> Generator[ComputationNode, None, None]:  # type: ignore
        yield self.generate_node()