"""Tests for the pluggable frontend boundary (plan/08).

- OnnxFrontend is bit-identical to the existing ONNXModelParser (the wrap doesn't diverge).
- The registry dispatches by ``can_load`` and raises for an unknown source.
- Import contract: no cost/solver/scheduler engine imports a concrete frontend module.
"""

from __future__ import annotations

import pathlib

import pytest

from stream.frontends import FrontendConfig, available_frontends, frontend_for, load_workload
from stream.frontends.onnx import OnnxFrontend
from stream.parser.onnx.model import ONNXModelParser

FIXTURES = [
    "stream/inputs/testing/workload/swiglu_1_16_32.onnx",
    "stream/inputs/testing/workload/attention_head.onnx",
    "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx",
]


def _signature(workload):
    """A structural fingerprint: (name, kind, operand-map strings) per node, in order."""
    sig = []
    for node in workload.nodes:
        maps = tuple(str(m) for m in getattr(node, "operand_mapping", ()))
        sig.append((node.name, getattr(node, "type", type(node).__name__), maps))
    return sig


@pytest.mark.parametrize("path", FIXTURES)
def test_onnx_frontend_is_bit_identical_to_the_parser(path: str):
    parser = ONNXModelParser(path)
    parser.run()
    via_frontend = OnnxFrontend().load(path, FrontendConfig())
    assert _signature(via_frontend) == _signature(parser.workload)


@pytest.mark.parametrize("path", FIXTURES)
def test_load_workload_dispatches_to_onnx(path: str):
    workload = load_workload(path)
    assert workload.get_computation_nodes()


def test_onnx_frontend_is_registered():
    assert any(f.name == "onnx" for f in available_frontends())


def test_can_load_rejects_non_onnx():
    onnx = OnnxFrontend()
    assert onnx.can_load("model.onnx")
    assert not onnx.can_load("model.pt")
    assert not onnx.can_load(42)


def test_frontend_for_raises_on_unknown_source():
    with pytest.raises(ValueError, match="no registered frontend"):
        frontend_for("model.unknownformat")


def test_no_engine_imports_a_concrete_frontend():
    """Import contract (plan/08, plan/10): engines depend on the protocol/registry, never on a concrete
    frontend module. A concrete frontend is imported only inside stream/frontends and the parsing stage."""
    root = pathlib.Path("stream")
    engine_dirs = [root / "opt", root / "cost_model", root / "stages" / "estimation", root / "stages" / "allocation"]
    offenders: list[str] = []
    for engine_dir in engine_dirs:
        for py in engine_dir.rglob("*.py"):
            text = py.read_text()
            if "stream.frontends" in text:
                offenders.append(str(py))
    assert offenders == [], f"engine modules must not import a frontend: {offenders}"
