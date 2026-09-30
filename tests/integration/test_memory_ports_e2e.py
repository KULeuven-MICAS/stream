"""The memory-ports family on the example ZigZag accelerators, against the prototype that derived it."""

import tempfile

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta

HARDWARE = "stream/inputs/examples/hardware/{}.yaml"
TWO_CONV = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)
SWIGLU_TILING = [
    {"dim": "Gemm_Left.D1", "tile": 128},
    {"dim": "Gemm_Down.D2", "tile": 128},
    {"dim": "Gemm_Left.D2", "tile": 32},
    {"dim": "Gemm_Left.D0", "tile": 16},
]

# Total latency and most utilised (memory, port) from the prototype run in evidence_59a9340/ports.jsonl.
TWO_CONV_CASES = [
    ("eyeriss_like_single_core", 114999, ("dram", "rw_port_1")),
    ("eyeriss_like_dual_core", 80915, ("sram_1M", "rw_port_2")),
    ("eyeriss_like_quad_core", 46031, ("sram_1M", "rw_port_2")),
    ("tpu_like_quad_core", 18072, ("dram", "rw_port_1")),
    ("simba_small", 14913, ("dram", "rw_port_1")),
    ("simba", 12308, ("dram", "rw_port_1")),
    ("fusemax", 185892, ("sram", "r_port_1")),
    ("meta_prototype_dual_core_simd_offchip", 21625, ("dram", "rw_port_1")),
]
SWIGLU_CASES = [
    ("eyeriss_like_single_core", 303038597, ("sram_64KB", "r_port_1")),
    ("eyeriss_like_dual_core", 201589115, ("dram", "rw_port_1")),
    ("eyeriss_like_quad_core", 201589494, ("dram", "rw_port_1")),
    ("tpu_like_quad_core", 201588814, ("dram", "rw_port_1")),
    ("simba_small", 201589342, ("dram", "rw_port_1")),
    ("simba", 201589014, ("dram", "rw_port_1")),
    ("fusemax", 76808227, ("sram", "r_port_1")),
    ("meta_prototype_dual_core_simd_offchip", 201588793, ("dram", "rw_port_1")),
]


def solve_with_ports(
    monkeypatch: pytest.MonkeyPatch, hardware: str, workload: str, **options: object
) -> tta.TransferAndTensorAllocator:
    solved: list[tta.TransferAndTensorAllocator] = []
    original = tta.TransferAndTensorAllocator.solve

    def capture(self: tta.TransferAndTensorAllocator, **kwargs: bool) -> object:
        solved.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(tta.TransferAndTensorAllocator, "solve", capture)
    with tempfile.TemporaryDirectory() as out:
        evaluate_mapping(
            HARDWARE.format(hardware), workload, out, options=SolveOptions(families=["memory_ports"], **options)
        )
    return solved[-1]


def busiest_port(alloc: tta.TransferAndTensorAllocator) -> tuple[str, str]:
    family = alloc.families[0]
    interval = alloc.model.value(alloc.quantities.get("iteration").expr) - alloc.overlap.X
    demand = alloc.quantities.indexed("port_demand")
    key = max(demand, key=lambda k: alloc.model.value(demand[k].expr) / (family.rate[k] * interval))
    return key[1], key[2]


@pytest.mark.slow
@pytest.mark.parametrize(("hardware", "latency", "port"), TWO_CONV_CASES, ids=[c[0] for c in TWO_CONV_CASES])
def test_two_conv_matches_the_prototype(
    monkeypatch: pytest.MonkeyPatch, hardware: str, latency: int, port: tuple[str, str]
) -> None:
    alloc = solve_with_ports(monkeypatch, hardware, make_2_conv_workload(TWO_CONV))
    assert alloc.total_latency.X == latency
    assert busiest_port(alloc) == port


@pytest.mark.slow
@pytest.mark.parametrize(("hardware", "latency", "port"), SWIGLU_CASES, ids=[c[0] for c in SWIGLU_CASES])
def test_fused_swiglu_matches_the_prototype(
    monkeypatch: pytest.MonkeyPatch, hardware: str, latency: int, port: tuple[str, str]
) -> None:
    workload = make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192)
    alloc = solve_with_ports(monkeypatch, hardware, workload, stage_options={"intra_core_tiling": SWIGLU_TILING})
    assert alloc.total_latency.X == latency
    assert busiest_port(alloc) == port
