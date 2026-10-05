"""Fused convolutions tiled along a sliding window: the oracle against the hand-derived sizes, and every solved
allocation against the oracle (bf16 elements; per core in the mapping's core order; interior iteration)."""

import pytest
from conv_windows import SCENARIOS, conv_chain, oracle, reads, solve

from stream.frontends import load_workload

C1, C2 = (1, 16, 32, 32), (1, 32, 32, 32)
SPEC = {
    "s1": {
        "iterations": 1,
        "conv1_tile": (C1,),
        "out_tile": (C2,),
        "input_staged": ((1, 8, 32, 32),),
        "input_moved": ((1, 8, 32, 32),),
        "window": (C1,),
        "moved": (C1,),
        "sources": {0: {0: 16384}},
        "conv1_rows": (32,),
        "input_moved_rows": (32,),
    },
    "s2": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 32),),
        "out_tile": ((1, 32, 8, 32),),
        "input_staged": ((1, 8, 10, 32),),
        "input_moved": ((1, 8, 8, 32),),
        "window": ((1, 16, 10, 32),),
        "moved": ((1, 16, 8, 32),),
        "sources": {0: {0: 4096}},
        "conv1_rows": (9, 8, 8, 7),
        "input_moved_rows": (10, 8, 8, 6),
    },
    "s3": {
        "iterations": 1,
        "conv1_tile": ((1, 16, 32, 8),) * 4,
        "out_tile": ((1, 32, 32, 8),) * 4,
        "input_staged": ((1, 8, 32, 9), (1, 8, 32, 10), (1, 8, 32, 10), (1, 8, 32, 9)),
        "input_moved": ((1, 8, 32, 9), (1, 8, 32, 10), (1, 8, 32, 10), (1, 8, 32, 9)),
        "window": ((1, 16, 32, 9), (1, 16, 32, 10), (1, 16, 32, 10), (1, 16, 32, 9)),
        "moved": ((1, 16, 32, 9), (1, 16, 32, 10), (1, 16, 32, 10), (1, 16, 32, 9)),
        "sources": {
            0: {0: 4096, 1: 512},
            1: {0: 512, 1: 4096, 2: 512},
            2: {1: 512, 2: 4096, 3: 512},
            3: {2: 512, 3: 4096},
        },
        "conv1_rows": (32,),
        "input_moved_rows": (32,),
    },
    "s4": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 16),) * 2,
        "out_tile": ((1, 32, 8, 16),) * 2,
        "input_staged": ((1, 8, 10, 17),) * 2,
        "input_moved": ((1, 8, 8, 17),) * 2,
        "window": ((1, 16, 10, 17),) * 2,
        "moved": ((1, 16, 8, 17),) * 2,
        "sources": {2: {0: 2048, 1: 128}, 3: {0: 128, 1: 2048}},
        "conv1_rows": (9, 8, 8, 7),
        "input_moved_rows": (10, 8, 8, 6),
    },
    "s5": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 32),),
        "out_tile": ((1, 32, 4, 16),),
        "input_staged": ((1, 8, 10, 32),),
        "input_moved": ((1, 8, 8, 32),),
        "window": ((1, 16, 9, 32),),
        "moved": ((1, 16, 8, 32),),
        "sources": {0: {0: 4096}},
        "conv1_rows": (8, 8, 8, 8),
        "input_moved_rows": (9, 8, 8, 7),
    },
}
CHECKS = tuple(SPEC["s1"])
IN_PLACE = {"s1": True, "s2": True, "s3": False, "s4": False, "s5": True, "s3_shared": True, "s4_shared": True}
RUNS = [
    *((s, hw) for s in ("s1", "s2", "s3", "s4", "s5") for hw in ("eyeriss_like_quad_core", "tpu_like_quad_core")),
    *((s, "simba_small") for s in ("s1", "s2", "s3", "s4", "s5")),
    *((s, "fusemax") for s in ("s1", "s2", "s5", "s3_shared", "s4_shared")),
]
SPLIT = {"s3", "s4", "s3_shared"}
XFAIL = {
    "input_staged": ("step 3: a transfer's footprint comes from its consumer's window", SPLIT),
    "window": ("step 3: a transfer's footprint comes from its consumer's window", set(SCENARIOS) - {"s1"}),
    "sources": ("step 3: routes pair cores whose tiles overlap", SPLIT),
    "in_place": ("step 3: routes pair cores whose tiles overlap", {"s3"}),
    "input_moved": ("step 4: a sliding window moves only its new rows", set(SCENARIOS) - {"s1"}),
    "moved": ("step 4: a sliding window moves only its new rows", SPLIT),
    "conv1_rows": ("step 5: first and last iterations are not modelled", set(SCENARIOS)),
    "input_moved_rows": ("step 5: first and last iterations are not modelled", set(SCENARIOS)),
}


def _case(scenario: str, hardware: str, check: str):
    reason, failing = XFAIL.get(check, ("", set()))
    marks = [pytest.mark.slow] if hardware == "simba_small" else []
    if scenario in failing:
        marks.append(pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(scenario, hardware, check, marks=marks)


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("scenario", SPEC)
def test_oracle_matches_the_hand_derived_sizes(scenario: str, check: str):
    assert oracle(scenario)[check] == SPEC[scenario][check]


@pytest.mark.parametrize(
    ("stride", "sizes"), [(1, [1, 3, 3, 3, 3, 8, 16, 32, 32, 32]), (2, [1, 2, 2, 3, 3, 3, 3, 8, 16, 16, 16, 32])]
)
def test_fused_axes_are_one_unique_dimension(stride: int, sizes: list[int]):
    workload = load_workload(conv_chain(stride))
    assert sorted(workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]) == sizes


@pytest.mark.parametrize(("boundary", "tile"), [(None, 1), ("first", 0), ("last", 3)])
def test_a_reader_footprint_is_its_window_and_the_producer_footprint_its_tile(boundary, tile: int):
    """An 8-row OY tile: conv2 reads the rows the oracle enumerates for that tile, conv1 writes 8 rows."""
    workload = load_workload(conv_chain(1))
    conv1, conv2 = workload.get_computation_nodes()
    sizes = {z: workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]}
    sizes[workload.get_dims(conv2)[2]] = 8
    mid = conv2.inputs[0]
    tile_out = {(0, k, y, x) for k in range(32) for y in range(8 * tile, 8 * tile + 8) for x in range(32)}
    rows = len({index[2] for index in reads(conv2, mid, tile_out)})
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, conv2, boundary) == (1, 16, rows, 32)
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, boundary=boundary) == (1, 16, 8, 32)
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, conv1, boundary) == (1, 16, 8, 32)


@pytest.mark.parametrize(
    ("scenario", "hardware", "check"), [_case(*run, check) for run in RUNS for check in (*CHECKS, "in_place")]
)
def test_allocation_matches_the_oracle(scenario: str, hardware: str, check: str):
    found = solve(scenario, hardware).get(check, NotImplementedError(check))
    if isinstance(found, Exception):
        raise found
    assert found == (IN_PLACE[scenario] if check == "in_place" else oracle(scenario)[check])
