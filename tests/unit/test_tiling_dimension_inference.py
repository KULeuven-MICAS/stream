"""Fast guard on the unique-dimension inference that tiling generation depends on.

Parsing a real multi-layer model (resnet18) must resolve a stable set of unique loop dimensions with
correct sizes; a regression here silently corrupts every downstream tiling and allocation. This is
the fast, MILP-free counterpart to the slow resnet CO tests in ``tests/test_resnet_patterns.py``.
"""

from __future__ import annotations

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.utils import affine_bounds

_RESNET18 = "stream/inputs/examples/workload/resnet18.onnx"


def test_resnet18_unique_dimension_inference():
    """Guard resnet18's affine dimension bookkeeping: the spatial pyramid (112->56->28->14->7), the
    channel widths (64/128/256/512) and the 7x7/3x3 kernels with 3 input channels all resolve, with a
    stable unique-dimension count and every size positive -- caught here before the (much slower) MILP
    allocation ever runs."""
    parser = ONNXModelParser(_RESNET18)
    parser.run()
    workload = parser.workload

    unique_dims, _ = workload.unique_dimensions()
    sizes = workload.get_dimension_sizes()

    assert len(unique_dims) == 71, f"Expected 71 unique loop dimensions, got {len(unique_dims)}"
    unique_sizes = [workload.get_dimension_size(z) for z in unique_dims]
    for node in workload.get_computation_nodes():
        for dim in workload.get_dims(node):
            low, high = affine_bounds(dim, unique_sizes)
            assert (low, high + 1) == (0, workload.get_dimension_size(dim)), f"{node.name}: {dim}"
    assert all(s > 0 for s in sizes), "Every inferred dimension size must be positive"

    distinct = set(sizes)
    assert {112, 56, 28, 14, 7}.issubset(distinct), f"Missing spatial pyramid extents in {sorted(distinct)}"
    assert {64, 128, 256, 512}.issubset(distinct), f"Missing channel widths in {sorted(distinct)}"
    assert {3, 7}.issubset(distinct), f"Missing input-channel / 7x7 stem-kernel extents in {sorted(distinct)}"
