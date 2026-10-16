from types import SimpleNamespace

from stream.compiler.kernels.library import CallDim
from stream.parser.mapping_factory import MappingFactory


class _Kernel:
    def __init__(self, *tile):
        self.tile = list(tile)

    def call_tile(self, node):
        return self.tile


class _Factory(MappingFactory):
    def __init__(self, nodes, extents):
        self.nodes, self.extents = nodes, extents
        self.workload = SimpleNamespace(get_dims=lambda node: node.dims, get_dimension_size=extents.__getitem__)

    def _group_kernels(self, layers):
        return [(self.nodes[name], self.nodes[name].kernel) for name in layers]


def _node(dims, kernel):
    return SimpleNamespace(dims=dims, kernel=kernel)


def test_the_default_tiling_is_the_call_tile_with_shared_dimensions_innermost():
    gemm = _node(("s", "e", "h"), _Kernel((1, 64, CallDim("k")), (2, 32, CallDim("n")), (0, 16, CallDim("m"))))
    silu = _node(("s", "h"), _Kernel((1, 32, CallDim("n", runtime=True)), (0, 16, CallDim("m", runtime=True))))
    factory = _Factory({"gemm": gemm, "silu": silu}, {"s": 256, "e": 128, "h": 512})
    assert factory._call_tile_tiling(("gemm", "silu")) == (("e", 64), ("h", 32), ("s", 16))


def test_a_whole_dimension_stays_only_when_the_kernel_keeps_it():
    silu = _node(("s", "h"), _Kernel((1, 64, CallDim("n", keep_whole=True)), (0, 256, CallDim("m"))))
    factory = _Factory({"silu": silu}, {"s": 256, "h": 64})
    assert factory._call_tile_tiling(("silu",)) == (("h", 64),)


def test_only_dimensions_every_kernel_sizes_at_runtime_may_grow():
    gemm = _node(("s", "h"), _Kernel((1, 32, CallDim("n")), (0, 16, CallDim("m", runtime=True))))
    silu = _node(("s", "h"), _Kernel((1, 32, CallDim("n", runtime=True)), (0, 16, CallDim("m", runtime=True))))
    factory = _Factory({"gemm": gemm, "silu": silu}, {"s": 256, "h": 512})
    assert factory._runtime_dims(("gemm", "silu"), (("s", 16), ("h", 32))) == ("s",)


def test_a_dimension_no_kernel_addresses_is_tiled_one_at_a_time_outermost():
    """Attention's heads lead every operand of a batched matmul, which its kernel is called
    once per head of; a unit axis needs no loop."""
    gemm = _node(
        ("u", "b", "s", "e", "h"), _Kernel((3, 64, CallDim("k")), (4, 32, CallDim("n")), (2, 16, CallDim("m")))
    )
    factory = _Factory({"gemm": gemm}, {"u": 1, "b": 4, "s": 256, "e": 128, "h": 512})
    assert factory._call_tile_tiling(("gemm",)) == (("e", 64), ("h", 32), ("s", 16), ("b", 1))
