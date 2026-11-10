"""A mapping the steady state cannot realise is refused before the solve, with a message naming what is wrong."""

import tempfile
from pathlib import Path

import pytest
from conv_windows import HARDWARE, conv, conv_chain, onnx_graph
from onnx import helper

from stream.api import SolveOptions, evaluate_mapping
from stream.frontends import load_workload
from stream.workload.iterator_type import NonlinearReductionUnrollError
from stream.workload.models import AttentionConfig, build_attention_block
from stream.workload.utils import UnfinishedReductionUnrollError, unfinished_reductions
from stream.workload.workload import determine_fusion_cut_points


def _mapping(splits: dict[str, int], cores: list[int], dim: str | None, tile: str) -> str:
    """The layers of ``splits`` fused in group g on ``cores``, each splitting ``dim`` by its factor, tiled ``tile``."""
    entries = "".join(
        f"- name: {name}\n  core_allocation: [{cores}]\n"
        + (f"  inter_core_tiling: [[{{dim: {dim}, split: {factor}}}]]\n" if factor > 1 else "")
        for name, factor in splits.items()
    )
    layers = ", ".join(splits)
    return f"layers:\n{entries}fused_groups:\n- name: g\n  layers: [{layers}]\n  intra_core_tiling: [{tile}]\n"


def _evaluate(workload: str, mapping: str, tile_search: bool = False) -> None:
    with tempfile.TemporaryDirectory() as out:
        (Path(out) / "m.yaml").write_text(mapping)
        options = SolveOptions(artifacts=False, tile_search=tile_search)
        evaluate_mapping(HARDWARE.format("simba_small"), workload, out, f"{out}/m.yaml", options)


def _odd_rows() -> str:
    """input (1,4,11,15) -> Conv1 (8, 3x3, pad 1) -> (1,8,11,15) -> Conv2 (8, 3x3, stride 2, pad 1) -> (1,8,6,8)."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out", stride=2)]
    return onnx_graph(nodes, {"input": [1, 4, 11, 15], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


def _rows(height: int, stride: int) -> str:
    """input (1,8,height,12) -> Conv1 (8, 3x3, pad 1) -> Conv2 (8, 3x3, ``stride``, pad 1)."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out", stride=stride)]
    return onnx_graph(nodes, {"input": [1, 8, height, 12], "w1": [8, 8, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


def _identity_skip(softmax: bool = False) -> str:
    """input (1,8,16,16) -> Conv1 (8, 3x3) -> Conv2 (8, 3x3) -> Add1 with input itself, with ``softmax`` then Norm, a
    softmax over the channels."""
    nodes = [
        conv("Conv1", "input", "w1", "c1"),
        conv("Conv2", "c1", "w2", "c2"),
        helper.make_node("Add", ["c2", "input"], ["s" if softmax else "out"], name="Add1"),
        *([helper.make_node("Softmax", ["s"], ["out"], name="Norm", axis=1)] if softmax else []),
    ]
    return onnx_graph(nodes, {"input": [1, 8, 16, 16], "w1": [8, 8, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


@pytest.mark.parametrize(
    ("workload", "cores", "split", "factor", "axis", "extent"),
    [
        (_odd_rows, [0, 1], "D2", 2, "D2", 11),
        (lambda: conv_chain(1), [0, 1, 2], "D6", 3, "D6", 16),
        (lambda: _rows(12, 2), [0, 1, 2, 3], "D2", 4, "D2 \\(extent 12, stride 2\\)", 6),
    ],
    ids=["11 rows over 2 cores", "16 channels over 3 cores", "12 rows a stride-2 reader walks over 4 cores"],
)
def test_a_core_split_that_does_not_divide_its_axis_names_node_dim_extent_and_factor(
    workload, cores: list[int], split: str, factor: int, axis: str, extent: int
):
    """Conv1's 11 output rows (same padding of 11 input rows) do not split over 2 cores, nor its 16 output channels
    over 3; its 12 rows, walked by a stride-2 reader as 2*z + r, split z, of extent 12 / 2 = 6, which 4 does not
    divide: the steady state has no remainder tiles, so the mapping is refused before the solve."""
    message = (
        f"Conv1 splits {axis} over {factor} cores, but the axis it splits, z\\d+, has extent {extent}, which {factor} "
        "does not divide."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        _evaluate(workload(), _mapping({"Conv1": factor, "Conv2": factor}, cores, split, "{dim: Conv2.D1, tile: 2}"))


@pytest.mark.parametrize("tile_search", [False, True], ids=["fixed tile", "tile search"])
def test_core_splits_of_one_dim_must_nest(tile_search: bool):
    """Conv2's 3 cores take 12 / 3 = 4 rows each, Conv1's 2 take 6: 2 does not divide 3, so Conv1's halves are not
    whole groups of Conv2's cores, though each split alone divides the 12 rows. A tile search over the mapping
    reports the same refusal, not that no tile candidate was priced."""
    message = (
        "Conv1 splits D2 over 2 cores and Conv2 splits D2 over 3, but 2 does not divide 3: splits of one dim "
        "\\(z\\d+\\) must nest."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        _evaluate(
            _rows(12, 1), _mapping({"Conv1": 2, "Conv2": 3}, [0, 1, 2], "D2", "{dim: Conv2.D2, tile: 2}"), tile_search
        )


def test_a_fused_tile_times_its_core_split_must_divide_the_axis():
    """Conv2's 32 rows tiled by 16 on each of 4 cores would need 16 x 4 = 64 rows; Conv1, first to split them by 4,
    is named."""
    message = (
        "Fused group g tiles z\\d+ by 16 and Conv1 splits it over 4 cores, but its extent 32 is not a multiple of 64."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        _evaluate(conv_chain(1), _mapping({"Conv1": 4, "Conv2": 4}, [0, 1, 2, 3], "D2", "{dim: Conv2.D2, tile: 16}"))


def test_tiling_a_reduction_a_later_node_walks_names_the_reducer_and_walkers():
    """The identity skip makes Conv1's 8 input channels (its D5, reduced) Add1's channels and Conv2's 8 output
    channels: tiling Conv2.D6 by 4 cuts that dim into 8 / 4 = 2 blocks, each needing Conv1's whole reduction."""
    message = (
        "Fused group g tiles z\\d+ into 2 blocks, but Conv1.D5 reduces over it and Add1, Conv2, computed from its "
        "output, walk it: each block would read Conv1's unfinished reduction. Keep z\\d+ whole or cut the group after "
        "Conv1."
    )
    with pytest.raises(UnfinishedReductionUnrollError, match=f"^{message}$") as refused:
        _evaluate(
            _identity_skip(), _mapping({"Conv1": 1, "Conv2": 1, "Add1": 1}, [0], None, "{dim: Conv2.D6, tile: 4}")
        )
    assert isinstance(refused.value, NonlinearReductionUnrollError)


def test_a_reduction_walked_later_and_normalized_is_refused_as_nonlinear():
    """Conv1 reduces its input channels (D5), which the skip makes Conv2's output channels, walked by Add1 and Norm;
    Norm, a softmax over the channels (its D1), also reduces them nonlinearly. The nonlinear reason is kept, since its
    remedy (the online-softmax rewrite) is the one that applies, and tiling Conv2.D6 by 4 into 8 / 4 = 2 blocks is
    refused with it."""
    workload = _identity_skip(softmax=True)
    why = next(iter(unfinished_reductions(load_workload(workload)).values()))
    assert (why.node, why.position, why.nonlinear) == ("Norm", 1, True)
    message = (
        "Fused group g tiles z\\d+ into 2 blocks, but Norm.D1 reduces nonlinearly over it "
        "\\(softmax/normalization\\): keep z\\d+ whole or apply the online-softmax rewrite."
    )
    with pytest.raises(NonlinearReductionUnrollError, match=f"^{message}$") as refused:
        _evaluate(
            workload,
            _mapping(dict.fromkeys(["Conv1", "Conv2", "Add1", "Norm"], 1), [0], None, "{dim: Conv2.D6, tile: 4}"),
        )
    assert type(refused.value) is NonlinearReductionUnrollError


def test_a_softmax_reduction_is_kept_whole_as_a_nonlinear_reduction_of_the_softmax():
    """The attention softmax reduces its key axis nonlinearly; each reduced axis is kept whole for that reason."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    group = next(
        sub
        for sub in workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))
        if any(n.type == "Softmax" for n in sub.get_computation_nodes())
    )
    softmax = next(n for n in group.get_computation_nodes() if n.type == "Softmax")
    reasons = unfinished_reductions(group)
    for position in softmax.reduction_axes:
        why = reasons[group.get_dims(softmax)[position]]
        assert (why.node, why.position, why.nonlinear) == (softmax.name, position, True)
