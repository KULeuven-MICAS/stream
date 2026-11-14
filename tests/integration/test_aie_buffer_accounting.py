"""The memory the allocation charges each tile is what mlir-aie allocates there for the design it generates."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest
import yaml

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
pytest.importorskip("aie.iron.kernels", reason="the library binds its kernels through mlir-aie")

from stream.api import SolveOptions, default_families, generate_code  # noqa: E402
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping  # noqa: E402
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping  # noqa: E402
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload  # noqa: E402
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload  # noqa: E402

ACCELERATOR = "stream/inputs/aie/hardware/whole_array_strix.yaml"
LIBRARY = "stream/inputs/aie/kernels/aie2p.toml"
FIXTURES = Path(__file__).parent.parent / "fixtures" / "aie"
ELEMENT_BYTES = {"bf16": 2, "f32": 4, "i32": 4}
BUFFER = re.compile(r"aie\.buffer\(%(?:mem_)?(tile_\d+_\d+)\)\s*\{[^}]*\}\s*:\s*memref<([^>]+)>")


def _aie_opt() -> str:
    found = shutil.which("aie-opt") or str(Path(sys.executable).parent / "aie-opt")
    if not Path(found).exists():
        pytest.skip("aie-opt is not installed")
    return found


def _allocated_bytes(final: Path) -> Counter[str]:
    """Per tile, the bytes of the buffers mlir-aie's object fifo lowering gives the design."""
    lowered = subprocess.run(
        [_aie_opt(), "--aie-objectFifo-stateful-transform", str(final)], capture_output=True, text=True, check=True
    ).stdout
    allocated: Counter[str] = Counter()
    for tile, memref in BUFFER.findall(lowered):
        *dims, element = memref.split("x")
        allocated[tile] += math.prod(map(int, dims)) * ELEMENT_BYTES[element]
    return allocated


def _charged_bytes(estimate) -> Counter[str]:
    occupancy = estimate.context.get("allocation").solution.performance["memory_occupancy"]
    return Counter({row["memory"]: row["resident_bits"] // 8 for row in occupancy if row["memory"].startswith("tile_")})


def _mha(out: str) -> tuple[str, str, SolveOptions]:
    workload, mapping = (str(FIXTURES / f"iron_mha_flash_causal_s256_d64.{ext}") for ext in ("onnx", "yaml"))
    return workload, mapping, SolveOptions(nb_cols_to_use=4, stage_options={"npu": "npu2"})


def _mha_halves(out: str) -> tuple[str, str, SolveOptions]:
    """The same attention at 64-query blocks, the value GEMM split by halves of the head on the rows either side of the
    softmax, ``matmul_PV`` built at a half."""
    workload = str(FIXTURES / "iron_mha_flash_causal_s256_d64.onnx")
    mapping = str(FIXTURES / "iron_mha_flash_causal_s256_d64_halves.yaml")
    library = tomllib.loads(Path(LIBRARY).read_text())
    pv = library["kernel"]["matmul_PV"]
    pv["dims"] = [{"name": "n", "blocks": [32, 64]} if d["name"] == "n" else d for d in pv["dims"]]
    return workload, mapping, SolveOptions(nb_cols_to_use=4, kernel_library=library, stage_options={"npu": "npu2"})


def _swiglu(out: str) -> tuple[str, str, SolveOptions]:
    workload = make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=False, output_dir=out)
    mapping = make_swiglu_mapping(256, 512, 2048, False, 32, 32, 64, output_dir=out)
    return workload, mapping, SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"})


def _swiglu_down(out: str) -> tuple[str, str, SolveOptions]:
    """The whole block fused, its down projection summing 8 output tiles per core across the hidden loop in float
    accumulators."""
    workload = make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=True, output_dir=out)
    mapping = make_swiglu_mapping(256, 512, 2048, True, 32, 32, 64, output_dir=out)
    return workload, mapping, SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"})


def _projection(out: str) -> tuple[str, str, SolveOptions]:
    """A projection whose memory tiles hold the activation's 64 x 2048 slab once across the column blocks and refill it
    per row block, beside the cores' float accumulators."""
    workload = make_gemm_workload(1024, 2048, 8192, "bf16", "bf16", out)
    mapping = make_gemm_mapping(1024, 2048, 8192, 64, 64, 64, nb_rows_to_use=4, nb_cols_to_use=8, output_dir=out)
    tiled = yaml.safe_load(Path(mapping).read_text())
    tiled["fused_groups"][0]["intra_core_tiling"] = [{"dim": f"Gemm.{d}", "tile": 64} for d in ("D1", "D2", "D0")]
    Path(mapping).write_text(yaml.safe_dump(tiled, sort_keys=False))
    overlap = {"overlap": {"transfer_contention": False, "offchip_contention": False}}
    families = default_families(ACCELERATOR, options=overlap)
    return workload, mapping, SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"}, families=families)


def _projection_replayed(out: str) -> tuple[str, str, SolveOptions]:
    """A projection whose memory tiles replay the activation's 64 x 1024 slab over the 8 column blocks and refill it
    per row block."""
    workload = make_gemm_workload(512, 1024, 4096, "bf16", "bf16", out)
    mapping = make_gemm_mapping(512, 1024, 4096, 64, 64, 64, nb_rows_to_use=4, nb_cols_to_use=8, output_dir=out)
    tiled = yaml.safe_load(Path(mapping).read_text())
    tiled["fused_groups"][0]["intra_core_tiling"] = [{"dim": f"Gemm.{d}", "tile": 64} for d in ("D1", "D2", "D0")]
    Path(mapping).write_text(yaml.safe_dump(tiled, sort_keys=False))
    overlap = {"overlap": {"transfer_contention": False, "offchip_contention": False}}
    families = default_families(ACCELERATOR, options=overlap)
    return workload, mapping, SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"}, families=families)


@pytest.fixture(
    scope="module",
    params=[_mha, _mha_halves, _swiglu, _swiglu_down, _projection, _projection_replayed],
    ids=["mha", "mha_halves", "swiglu", "swiglu_down", "projection", "projection_replayed"],
)
def accounts(request, tmp_path_factory) -> tuple[str, Counter[str], Counter[str], Path]:
    """A design's name, the bytes the allocation charges each tile, the bytes mlir-aie allocates there, and its
    ``final.mlir``."""
    out = tmp_path_factory.mktemp(request.param.__name__)
    workload, mapping, options = request.param(str(out))
    estimate = generate_code(ACCELERATOR, workload, str(out), mapping, options)
    (final,) = out.glob("**/codegen/final.mlir")
    return request.param.__name__, _charged_bytes(estimate), _allocated_bytes(final), final


@pytest.mark.slow
def test_no_tile_holds_more_than_it_is_charged(accounts):
    _, charged, allocated, _ = accounts
    assert allocated
    assert {tile: (charged[tile], held) for tile, held in allocated.items() if held > charged[tile]} == {}


@pytest.mark.slow
def test_mlir_aie_gives_every_object_fifo_its_buffers_and_descriptors(accounts):
    """The pool and endpoint lowering, which walks a memory tile's pools whole where it runs short of descriptors."""
    *_, final = accounts
    passes = ["--aie-objectfifo-split", "--aie-objectfifo-verify", "--aie-objectfifo-allocate"]
    lowered = subprocess.run([_aie_opt(), *passes, str(final)], capture_output=True, text=True, check=False)
    assert lowered.returncode == 0, lowered.stderr[:2000]


@pytest.mark.slow
def test_each_tile_is_charged_the_buffers_mlir_aie_allocates(accounts, request):
    design, charged, allocated, _ = accounts
    if design in ("_swiglu", "_swiglu_down"):
        reason = (
            "the allocation sizes a tile of the element-wise cores over a spatiotemporal loop that codegen streams a "
            "kernel tile at a time, so it charges them more than they hold"
        )
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    assert charged == allocated
