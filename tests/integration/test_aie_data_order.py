"""A design's DMAs put each element where its cores read it and write it: the activation a memory tile stages for a row
of cores and the outputs it gathers from them, walked as the generated BDs walk them."""

from __future__ import annotations

import re
from itertools import product
from math import prod
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
pytest.importorskip("aie.iron.kernels", reason="the library binds its kernels through mlir-aie")

from stream.api import SolveOptions, default_families, generate_code  # noqa: E402
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping  # noqa: E402
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload  # noqa: E402

ACCELERATOR = "stream/inputs/aie/hardware/whole_array_strix.yaml"
M, K, N = 512, 1024, 512
CALL_ROWS, HELD = 32, 4
ROWS, COLUMNS = 4, 8
BLOCK = HELD * CALL_ROWS * ROWS
"""A core holds four 32-row output tiles across the reduction, and the four rows of cores split each 512-row block
with the core row innermost: tile ``j`` of core row ``r`` is rows ``j * 128 + r * 32`` on."""


@pytest.fixture(scope="module")
def design(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("rows_held")
    workload = make_gemm_workload(M, K, N, "bf16", "bf16", str(out))
    mapping = make_gemm_mapping(
        M, K, N, CALL_ROWS, 64, 64, nb_rows_to_use=ROWS, nb_cols_to_use=COLUMNS, output_dir=str(out)
    )
    tiled = yaml.safe_load(Path(mapping).read_text())
    tiling = [("D0", CALL_ROWS), ("D1", 64), ("D1", 512), ("D2", 64), ("D0", HELD * CALL_ROWS)]
    tiled["fused_groups"][0]["intra_core_tiling"] = [{"dim": f"Gemm.{d}", "tile": t} for d, t in tiling]
    Path(mapping).write_text(yaml.safe_dump(tiled, sort_keys=False))
    overlap = {"overlap": {"transfer_contention": False, "offchip_contention": False}}
    options = SolveOptions(
        nb_cols_to_use=COLUMNS,
        stage_options={"npu": "npu2"},
        families=default_families(ACCELERATOR, options=overlap),
        artifacts=False,
    )
    generate_code(ACCELERATOR, workload, str(out), mapping, options)
    (final,) = out.glob("**/codegen/final.mlir")
    return final.read_text()


def _ints(text: str) -> list[int]:
    return [int(v) for v in re.findall(r"-?\d+", text)]


def _shim_bds(design: str, fifo: str) -> list[tuple[int, list[int], list[int]]]:
    """Each BD the runtime sequence configures for ``fifo``, in order: its offset, sizes and strides, outer first."""
    bds = []
    pattern = (
        rf"alloc = @{fifo}, [^\n]*\n\s*\"aie\.dma_bd\"\(%\d+\) <\{{static_offset = (\d+)[^}}]*"
        r"static_sizes = array<i64: ([\d, ]+)>, static_strides = array<i64: ([\d, ]+)>"
    )
    for offset, sizes, strides in re.findall(pattern, design):
        bds.append((int(offset), _ints(sizes), _ints(strides)))
    return bds


def _walk(offset: int, sizes: list[int], strides: list[int]) -> np.ndarray:
    """The addresses a walk visits, in order."""
    return np.array(
        [offset + sum(i * s for i, s in zip(index, strides, strict=True)) for index in product(*map(range, sizes))]
    )


def _fifo(design: str, name: str) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
    """The walk a memory tile writes ``name``'s objects with, as (size, stride) pairs outer first, and the object's
    shape."""
    line = next(line for line in design.splitlines() if f"aie.objectfifo @{name}(" in line)
    shape = tuple(_ints(re.search(r"memref<([\dx]+)xbf16>", line).group(1).replace("x", " ")))
    found = re.search(r"dimensionsFromStream \[([^\]]*)\]", line)
    pairs = re.findall(r"size = (\d+), stride = (\d+)", found.group(1)) if found else []
    return [(int(a), int(b)) for a, b in pairs], shape


@pytest.mark.slow
def test_a_memory_tile_stages_each_held_row_tile_where_its_core_row_reads_it(design):
    """The first object row ``r``'s memory tile holds is, k tile by k tile, the four 32-row tiles that row of cores
    calls on, at their rows of the activation."""
    for r in range(ROWS):
        walk, shape = _fifo(design, f"of_0_mem_{r}")
        offset, sizes, strides = _shim_bds(design, f"of_0_mem_{r}")[0]
        assert sizes[-1] == shape[0] * shape[-1], "the shim reads a row of the staged block in one run"
        stream = _walk(offset, sizes, strides)[: prod(shape)]
        held = np.empty(prod(shape), dtype=np.int64)
        held[_walk(0, [s for s, _ in walk], [t for _, t in walk]) if walk else np.arange(prod(shape))] = stream
        k_tiles, tiles, rows, columns = shape
        held = held.reshape(shape)
        for kt, j in product(range(k_tiles), range(tiles)):
            first = (j * CALL_ROWS * ROWS + r * CALL_ROWS) * K + kt * columns
            expected = first + np.add.outer(np.arange(rows) * K, np.arange(columns))
            assert (held[kt, j] == expected).all(), (r, kt, j)


@pytest.mark.slow
def test_a_memory_tile_gathers_one_round_of_its_cores_output_tiles_at_a_time(design):
    """Each object the memory tile gathers holds one output tile of each core row, so the row-major run it writes off
    chip puts tile ``j`` of core row ``r`` at rows ``j * 128 + r * 32`` on."""
    link = re.search(r"aie\.objectfifo\.link \[([^\]]*)\] -> \[@(of_\d+_mem_0)\]\(\[([\d, ]*)\]", design)
    sources, gathered, offsets = link.group(1).split(", "), link.group(2), _ints(link.group(3))
    _, tile = _fifo(design, sources[0].lstrip("@"))
    _, shape = _fifo(design, gathered)
    assert prod(shape) == len(sources) * prod(tile)
    assert offsets == [r * prod(tile) for r in range(len(sources))]
    offset, sizes, strides = _shim_bds(design, gathered)[0]
    rows = _walk(offset, sizes, strides) // N
    per_object = rows[: HELD * prod(shape)].reshape(HELD, len(sources), CALL_ROWS, -1)
    for j, r in product(range(HELD), range(len(sources))):
        assert per_object[j, r, :, 0].tolist() == list(
            range(j * BLOCK // HELD + r * CALL_ROWS, j * BLOCK // HELD + (r + 1) * CALL_ROWS)
        )
