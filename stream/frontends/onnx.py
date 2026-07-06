"""The built-in ONNX frontend: a thin, bit-identical wrap over the existing ONNX parser.

``OnnxFrontend.load`` produces exactly the workload :class:`~stream.parser.onnx.model.ONNXModelParser`
already produces -- it is the sanctioned ingestion seam, not a reimplementation. ONNX stays the default,
most-validated path; the frontend protocol just makes it one plugin among future ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stream.frontends import FrontendConfig, register_frontend

if TYPE_CHECKING:
    from stream.workload.workload import Workload


class OnnxFrontend:
    """Loads an ONNX model from a ``.onnx`` path into a workload (what ``ONNXModelParser`` accepts)."""

    name = "onnx"

    def can_load(self, source: Any) -> bool:
        return isinstance(source, str) and source.endswith(".onnx")

    def load(self, source: Any, config: FrontendConfig | None = None) -> Workload:
        from stream.parser.onnx.model import ONNXModelParser  # noqa: PLC0415

        parser = ONNXModelParser(source)
        parser.run()
        return parser.workload


# Register the built-in so `import stream.frontends.onnx` makes ONNX available even without entry points.
register_frontend(OnnxFrontend())
