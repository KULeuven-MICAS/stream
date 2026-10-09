"""The generic mapper fuses each ResNet-18 residual block, streams its activations along OX, splits it over the cores
by what suits the block (rows where the weights fit, channels where they do not) and cuts a block only where its
weights cannot stay on chip."""

import tempfile

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

_RESNET18 = "stream/inputs/examples/workload/resnet18.onnx"
_HARDWARE = "stream/inputs/examples/hardware/{}.yaml"
_PARTS = ("tpu_like_quad_core", "fusemax", "simba_small")


def _plan(part: str) -> tuple[GenericMappingGenerator, list]:
    ctx = StageContext.from_kwargs(
        accelerator=_HARDWARE.format(part), workload_path=_RESNET18, output_path=tempfile.mkdtemp()
    )
    ctx = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0]
    workload = ctx.get("workload")
    generator = GenericMappingGenerator(ctx.get("accelerator"), workload, tempfile.mkdtemp())
    return generator, workload.split_fusion_groups(cut_points=generator._cut_points(None))


def _short(group) -> list[str]:
    """Node names without the stage prefix: ``layer1.0/conv1/Conv``, ``conv1/Conv`` for the stem."""
    return [
        cn.name.removeprefix("/").split("/", 1)[-1] if cn.name.startswith("/layer") else cn.name[1:]
        for cn in group.get_computation_nodes()
    ]


@pytest.mark.parametrize("part", _PARTS)
def test_every_conv_group_streams_the_innermost_axis_of_its_activation(part):
    generator, groups = _plan(part)
    for group in groups:
        cns = tuple(group.get_computation_nodes())
        if not any(cn.type == "Conv" for cn in cns) or len(cns) == 1:
            continue
        if not generator._sliding_dims(group, cns) - set(generator._inter_core_unrolling(group, cns)):
            continue  # its cores split every axis a window slides along, so it streams its channels
        _, indexed = generator._indexed_by_intermediates(group, cns)
        axis = generator._streaming_axis(group, cns, set(indexed))
        assert generator._output_axis(group, cns, axis) == 3, _short(group)


@pytest.mark.parametrize("part", _PARTS)
def test_the_first_blocks_stay_fused_whole(part):
    _, groups = _plan(part)
    blocks = [_short(g) for g in groups]
    assert ["conv1/Conv", "relu/Relu", "maxpool/MaxPool"] in blocks
    for block in ("layer1.0", "layer1.1"):
        assert [f"{block}/conv1/Conv", f"{block}/relu/Relu", f"{block}/conv2/Conv"] == [
            name for g in blocks for name in g if name.startswith(f"{block}/conv") or name.startswith(f"{block}/relu/")
        ]
        assert any(len(g) == 5 and g[0] == f"{block}/conv1/Conv" for g in blocks)


def test_a_block_whose_weights_overflow_is_cut_between_its_convs():
    _, groups = _plan("tpu_like_quad_core")
    blocks = [_short(g) for g in groups]
    assert ["layer4.1/conv1/Conv", "layer4.1/relu/Relu"] in blocks
    assert ["layer4.1/conv2/Conv", "layer4.1/Add", "layer4.1/relu_1/Relu"] in blocks


@pytest.mark.parametrize(
    ("part", "block", "dim"),
    [("tpu_like_quad_core", "layer1.0", "D2"), ("simba_small", "layer1.0", "D2"), ("simba_small", "layer2.0", "D6")],
)
def test_a_block_splits_rows_where_its_weights_fit_and_channels_where_they_do_not(part, block, dim):
    generator, groups = _plan(part)
    group = next(g for g in groups if _short(g)[0] == f"{block}/conv1/Conv")
    conv = group.get_computation_nodes()[0]
    cores = generator._select_cores_for_node(conv)
    split = generator._factor_split_across_dims(group, conv, len(cores), generator._protected_dims(group, (conv,)))
    assert [f"D{idx}" for idx, _ in split] == [dim]


def test_the_stem_tiles_its_strided_window_in_whole_strides():
    generator, groups = _plan("tpu_like_quad_core")
    stem = groups[0]
    tiling = generator._build_intra_core_tiling(stem, tuple(stem.get_computation_nodes()))
    assert tiling
    for entry in tiling:
        node = entry["dim"].split(".D")[0]
        assert node == "/maxpool/MaxPool" or entry["tile"] % 2 == 0, tiling


@pytest.mark.slow
@pytest.mark.parametrize("part", _PARTS)
def test_resnet18_solves_end_to_end(part):
    estimate = evaluate_mapping(
        _HARDWARE.format(part), _RESNET18, tempfile.mkdtemp(), None, SolveOptions(nb_cols_to_use=4, backend="gurobi")
    )
    assert estimate.group_cycles and all(cycles > 0 for cycles in estimate.group_cycles)
