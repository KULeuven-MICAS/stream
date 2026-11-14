"""The memory the allocation charges each tile is what mlir-aie allocates there for the design it generates."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
pytest.importorskip("aie.iron.kernels", reason="the library binds its kernels through mlir-aie")

from stream.api import SolveOptions, generate_code  # noqa: E402
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping  # noqa: E402
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload  # noqa: E402

ACCELERATOR = "stream/inputs/aie/hardware/whole_array_strix.yaml"
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


def _swiglu(out: str) -> tuple[str, str, SolveOptions]:
    workload = make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=False, output_dir=out)
    mapping = make_swiglu_mapping(256, 512, 2048, False, 32, 32, 64, output_dir=out)
    return workload, mapping, SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"})


@pytest.fixture(scope="module", params=[_mha, _swiglu], ids=["mha", "swiglu"])
def accounts(request, tmp_path_factory) -> tuple[str, Counter[str], Counter[str]]:
    """A design's name, the bytes the allocation charges each tile and the bytes mlir-aie allocates there."""
    out = tmp_path_factory.mktemp(request.param.__name__)
    workload, mapping, options = request.param(str(out))
    estimate = generate_code(ACCELERATOR, workload, str(out), mapping, options)
    (final,) = out.glob("**/codegen/final.mlir")
    return request.param.__name__, _charged_bytes(estimate), _allocated_bytes(final)


@pytest.mark.slow
def test_no_tile_holds_more_than_it_is_charged(accounts):
    _, charged, allocated = accounts
    assert allocated
    assert {tile: (charged[tile], held) for tile, held in allocated.items() if held > charged[tile]} == {}


@pytest.mark.slow
def test_each_tile_is_charged_the_buffers_mlir_aie_allocates(accounts, request):
    design, charged, allocated = accounts
    if design == "_swiglu":
        reason = (
            "the allocation sizes a tile of the element-wise cores over a spatiotemporal loop that codegen streams a "
            "kernel tile at a time, so it charges them more than they hold"
        )
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    assert charged == allocated
