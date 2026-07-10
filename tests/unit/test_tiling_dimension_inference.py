"""Fast guard on the unique-dimension inference that tiling generation depends on.

Parsing a real multi-layer model (resnet18) must resolve a stable set of unique loop dimensions with
correct sizes; a regression here silently corrupts every downstream tiling and allocation. This is
the fast, MILP-free counterpart to the slow resnet CO tests in ``tests/test_resnet_patterns.py``.
"""

from __future__ import annotations

from stream.parser.onnx.model import ONNXModelParser

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

    # Every node dim resolves to a pure, tileable LayerDim: strided conv/pool axes are their own free
    # variable (not folded into an untileable compound), so the basis is wider than a naive identity
    # merge but every dimension is addressable.
    from stream.datatypes import LayerDim

    assert len(unique_dims) == 87, f"Expected 87 unique loop dimensions, got {len(unique_dims)}"
    assert all(
        isinstance(workload.get_dims(n)[i], LayerDim)
        for n in workload.get_computation_nodes()
        for i in range(len(workload.get_dims(n)))
    ), "Every node iteration dim must be a pure LayerDim (no compound expressions)"
    assert all(s > 0 for s in sizes), "Every inferred dimension size must be positive"

    distinct = set(sizes)
    assert {112, 56, 28, 14, 7}.issubset(distinct), f"Missing spatial pyramid extents in {sorted(distinct)}"
    assert {64, 128, 256, 512}.issubset(distinct), f"Missing channel widths in {sorted(distinct)}"
    assert {3, 7}.issubset(distinct), f"Missing input-channel / 7x7 stem-kernel extents in {sorted(distinct)}"
