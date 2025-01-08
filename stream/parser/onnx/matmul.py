import logging

from stream.parser.onnx.gemm import GemmParser

logger = logging.getLogger(__name__)


class MatMulParser(GemmParser):
    """! Parses an ONNX MatMul operator into a ComputationNode. Exactly the same as Gemm Parser"""
