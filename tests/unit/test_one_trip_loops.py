"""A fused tile that a core split already covers leaves no loop to step.

The flash attention's value GEMM split by halves of its 64-wide head over two cores takes one 32-wide tile on each:
a fused loop over the head would run once. Kept, it is a loop the output's buffer depends on, nested inside the key
loop the output is reused over, so code generation acquires and sends on an output block per key block.
"""

from __future__ import annotations

import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("aie.iron.kernels", reason="the library binds mha.cc through mlir-aie")

from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.datatypes import LayerDim  # noqa: E402
from stream.parser.mapping_parser import MappingParser  # noqa: E402
from stream.stages.context import StageContext  # noqa: E402
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage  # noqa: E402
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage  # noqa: E402
from stream.stages.stage import LeafStage, MainStage  # noqa: E402
from stream.workload.utils import determine_fusion_splits  # noqa: E402

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/aie"
MHA = "iron_mha_flash_causal_s256_d64"
# Score GEMM and softmax on four columns of one row each, the value GEMM split by halves of the head on two rows.
PLACEMENT = {
    "Attn_Scores": ((2,), ((1, 4),)),
    "Attn_Softmax": ((4,), ((1, 4),)),
    "Attn_Context": ((3, 5), ((2, 2), (1, 4))),
}


def _split_by_halves():
    """The MHA fixture placed by hand as ``PLACEMENT``, ``matmul_PV`` a half of the head a call."""
    data = tomllib.loads((ROOT / "stream/inputs/aie/kernels/aie2p.toml").read_text())
    pv = data["kernel"]["matmul_PV"]
    pv["dims"] = [{"name": "n", "blocks": [32, 64]} if d["name"] == "n" else d for d in pv["dims"]]
    with tempfile.TemporaryDirectory() as out:
        ctx = StageContext.from_kwargs(
            accelerator=str(ROOT / "stream/inputs/aie/hardware/whole_array_strix.yaml"),
            workload_path=str(FIXTURES / f"{MHA}.onnx"),
            output_path=out,
            kernel_library=KernelLibrary.from_dict(data),
        )
        ctx = next(iter(MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()))
    workload, accelerator = ctx.get("workload"), ctx.get("accelerator")
    mapping = MappingParser(str(FIXTURES / f"{MHA}.yaml"), workload, accelerator).run()
    grid = {(c.col_id, c.row_id): c for c in accelerator.core_list if c.type == "compute"}
    for name, (rows, split) in PLACEMENT.items():
        nm = mapping.get(workload.get_node_by_name(name))
        nm.resource_allocation = (tuple(grid[(col, row)] for row in rows for col in range(4)),)
        nm.inter_core_tiling = (tuple((LayerDim(position=p, prefix="d"), f) for p, f in split),)
    context = mapping.get(workload.get_node_by_name("Attn_Context"))
    context.kernel = replace(context.kernel, n=32)
    return workload, mapping


def test_a_tile_a_core_split_covers_leaves_no_loop_of_one_trip():
    workload, mapping = _split_by_halves()
    head = workload.get_dims(workload.get_node_by_name("Attn_Context"))[2]

    def splits(tile: int) -> dict:
        return determine_fusion_splits(
            workload, mapping.with_fused_groups([replace(mapping.fused_groups[0], intra_core_tiling=((head, tile),))])
        )

    assert head not in splits(32)
    assert splits(16)[head] == 2
