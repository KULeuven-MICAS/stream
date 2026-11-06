"""Stream cores and a reference node for the AccelForge tests."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import i8
from xdsl.ir.affine import AffineMap
from zigzag.utils import open_yaml

from stream.hardware.architecture.core import Core
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor

CORES = Path(__file__).parents[3] / "stream/inputs/examples/hardware/cores"
COMPUTE_CORES = (
    "eyeriss_like",
    "fusemax_array",
    "fusemax_vec",
    "meta_prototype",
    "pooling",
    "simba_chiplet",
    "simd",
    "tpu_like",
    "tpu_v7_mxu",
    "tpu_v7_vpu",
)


def load_core(source: str | Path | Mapping[str, Any]) -> Core:
    """A Stream core from a core YAML file or its data, validated and built by Stream's own parser."""
    data = copy.deepcopy(dict(source)) if isinstance(source, Mapping) else open_yaml(str(source))
    validator = AcceleratorValidator({}, "")
    normalized = validator.validate_core_data(data, str(data.get("name", "core")))
    assert normalized is not None and validator.is_valid, validator.errors
    return AcceleratorFactory({}).create_core(normalized, 0)


def example_core(name: str) -> Core:
    return load_core(CORES / f"{name}.yaml")


def matmul(size: int = 64) -> ComputationNode:
    """An 8-bit ``size``-cubed matmul, C[m][n] += A[m][k] * B[k][n]."""
    a, b, c = (Tensor.create(name, i8, (size, size)) for name in "ABC")
    maps = tuple(
        AffineMap.from_callable(fn) for fn in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
    )
    return ComputationNode(type="MatMul", name="mm", inputs=(a, b), outputs=(c,), operand_mapping=maps)


def conv(stride: int, window: int) -> ComputationNode:
    """I[c][oy + fy][stride * ox + fx] over 8x8 outputs and a ``window``-wide kernel along x."""
    x = Tensor.create("I", i8, (4, 10, stride * 7 + window))
    w, o = Tensor.create("W", i8, (8, 4, 3, window)), Tensor.create("O", i8, (8, 8, 8))
    maps = tuple(
        AffineMap.from_callable(fn)
        for fn in (
            lambda k, c, oy, ox, fy, fx: (c, oy + fy, stride * ox + fx),
            lambda k, c, oy, ox, fy, fx: (k, c, fy, fx),
            lambda k, c, oy, ox, fy, fx: (k, oy, ox),
        )
    )
    return ComputationNode(type="Conv", name="conv", inputs=(x, w), outputs=(o,), operand_mapping=maps)
