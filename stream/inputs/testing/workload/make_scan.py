"""A tiny Mamba1-style selective-scan workload, built directly on the affine IR.

The recurrence is a prefix sum ``h[t] = h[t-1] + x[t]`` over the sequence dimension ``t`` (element-
wise over the hidden dimension ``d``). The state operand ``h_prev`` is read at ``t-1`` and the
output ``h`` is written at ``t``, so ``t`` is a SEQUENTIAL iteration dimension (see
``stream.workload.iterator_type``). The op kind is deliberately simple; the rewrite library supplies
the real Mamba/SSD/DeltaNet math. ``scan_reference`` is the NumPy golden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xdsl.dialects.builtin import FixedBitwidthType, bf16
from xdsl.ir.affine import AffineMap

from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload


@dataclass(frozen=True)
class ScanConfig:
    seq_len: int = 8
    hidden: int = 16
    dtype: FixedBitwidthType = bf16


def make_scan_workload(config: ScanConfig | None = None) -> Workload:
    """Build the single-node scan workload: inputs (x, h_prev), output h; dims (t, d)."""
    config = config or ScanConfig()
    seq_len, hidden = config.seq_len, config.hidden
    x = Tensor.create("x", config.dtype, (seq_len, hidden))
    h_prev = Tensor.create("h_prev", config.dtype, (seq_len, hidden))
    h = Tensor.create("h", config.dtype, (seq_len, hidden))
    operand_mapping = (
        AffineMap.from_callable(lambda t, d: (t, d)),  # x[t, d]
        AffineMap.from_callable(lambda t, d: (t - 1, d)),  # h_prev[t-1, d] -- the state read
        AffineMap.from_callable(lambda t, d: (t, d)),  # h[t, d] -- the state written
    )
    scan = ComputationNode(type="Scan", name="scan", inputs=(x, h_prev), outputs=(h,), operand_mapping=operand_mapping)
    nodes = [
        InEdge(name="x", outputs=(x,)),
        InEdge(name="h_prev", outputs=(h_prev,)),
        scan,
        OutEdge(name="h", inputs=(h,)),
    ]
    return Workload(nodes)


def scan_reference(x: np.ndarray) -> np.ndarray:
    """Golden for ``h[t] = h[t-1] + x[t]`` with ``h[-1] = 0`` -- a prefix sum along the sequence axis."""
    return np.cumsum(x, axis=0)
