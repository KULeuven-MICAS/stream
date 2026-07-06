"""The ONNX parser registration seam -- how an out-of-tree package adds higher-level-op parsers
without forking (the same way it adds decompositions and blocks)."""

from __future__ import annotations

from stream.parser.onnx.model import _REGISTERED_PARSERS, ONNXModelParser, onnx_parser_for, register_onnx_parser
from stream.parser.onnx.operator_parser import OnnxOperatorParser


class _DummyParser(OnnxOperatorParser):
    def generate_node(self, name_to_tensor_dict):  # pragma: no cover - not exercised, only dispatched
        raise NotImplementedError


def test_builtin_parsers_are_looked_up():
    assert onnx_parser_for("MatMul") is not None
    assert onnx_parser_for("NoSuchOp") is None


def test_registers_a_higher_level_parser_and_it_dispatches():
    register_onnx_parser("MaskedAttention", _DummyParser)
    try:
        assert onnx_parser_for("MaskedAttention") is _DummyParser
        node = type("Node", (), {"op_type": "MaskedAttention"})()
        assert ONNXModelParser("x.onnx").get_parser_class(node) is _DummyParser
    finally:
        _REGISTERED_PARSERS.pop("MaskedAttention", None)


def test_an_overlay_registration_overrides_the_builtin_table():
    register_onnx_parser("Gemm", _DummyParser)  # override a built-in op type
    try:
        assert onnx_parser_for("Gemm") is _DummyParser
    finally:
        _REGISTERED_PARSERS.pop("Gemm", None)
    assert onnx_parser_for("Gemm") is not _DummyParser  # built-in restored after cleanup
