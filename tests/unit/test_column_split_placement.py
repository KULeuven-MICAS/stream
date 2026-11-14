"""A stacked chain's last layer, offered on twice its rows split over its output columns.

On the flash attention the value GEMM is the costliest layer and the softmax sends it a state per query block, so a
second row of value cores can only take a half of the head each, not a share of the query blocks. Nothing but the
placement offers that design, so its shape is pinned here.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("aie.iron.kernels", reason="the library binds mha.cc through mlir-aie")

from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.parser.mapping_parser import MappingParser  # noqa: E402
from stream.stages.context import StageContext  # noqa: E402
from stream.stages.generation.placement_generation import PlacementGenerationStage  # noqa: E402
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage  # noqa: E402
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage  # noqa: E402
from stream.stages.stage import LeafStage, MainStage  # noqa: E402

ROOT = Path(__file__).parents[2]
ACCELERATOR = ROOT / "stream/inputs/aie/hardware/whole_array_strix.yaml"
FIXTURES = ROOT / "tests/fixtures/aie"
MHA = "iron_mha_flash_causal_s256_d64"


def _library(head_blocks: tuple[int, ...]) -> KernelLibrary:
    """Stream's own library, ``matmul_PV`` building the heads ``head_blocks``."""
    data = tomllib.loads((ROOT / "stream/inputs/aie/kernels/aie2p.toml").read_text())
    dims = data["kernel"]["matmul_PV"]["dims"]
    data["kernel"]["matmul_PV"]["dims"] = [
        {"name": "n", "blocks": list(head_blocks)} if d["name"] == "n" else d for d in dims
    ]
    return KernelLibrary.from_dict(data)


def _placed(head_blocks: tuple[int, ...]):
    """The MHA fixture's mapping and the alternatives the placement offers, under ``_library(head_blocks)``."""
    with tempfile.TemporaryDirectory() as out:
        ctx = StageContext.from_kwargs(
            accelerator=str(ACCELERATOR),
            workload_path=str(FIXTURES / f"{MHA}.onnx"),
            output_path=out,
            kernel_library=_library(head_blocks),
        )
        ctx = next(iter(MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()))
        workload, accelerator = ctx.get("workload"), ctx.get("accelerator")
        mapping = MappingParser(str(FIXTURES / f"{MHA}.yaml"), workload, accelerator).run()
        ctx = StageContext.from_kwargs(workload=workload, mapping=mapping, accelerator=accelerator)
        placed = next(iter(PlacementGenerationStage([LeafStage], ctx).run()))
    return workload, placed.get("placement_alternatives")


def _layer(workload, mapping, name):
    nm = mapping.get(workload.get_node_by_name(name))
    cores = [(core.col_id, core.row_id) for core in nm.resource_allocation[0]]
    return cores, [(d.position, s) for d, s in nm.inter_core_tiling[0]], nm.kernel


def _column_split(workload, alternatives):
    found = [m for m in alternatives if len(_layer(workload, m, "Attn_Context")[0]) == 8]
    assert len(found) == 1, [_layer(workload, m, "Attn_Context")[:2] for m in alternatives]
    return found[0]


def test_the_value_gemm_is_offered_split_by_halves_of_the_head_around_the_softmax():
    workload, alternatives = _placed((32, 64))
    split = _column_split(workload, alternatives)
    score, softmax, context = (
        _layer(workload, split, name) for name in ("Attn_Scores", "Attn_Softmax", "Attn_Context")
    )
    columns = range(4)
    assert score[:2] == ([(c, 2) for c in columns], [(1, 4)])
    assert softmax[:2] == ([(c, 4) for c in columns], [(1, 4)])
    # The halves outermost: core j of each half holds query block j, as the softmax core j does.
    assert context[0] == [(c, 3) for c in columns] + [(c, 5) for c in columns]
    assert context[1] == [(2, 2), (1, 4)]
    assert context[2].n == 32


def test_no_column_split_where_the_library_builds_no_half_of_the_head():
    workload, alternatives = _placed((64,))
    assert all(len(_layer(workload, m, "Attn_Context")[0]) == 4 for m in alternatives)
