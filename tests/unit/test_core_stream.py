"""A core calls a kernel no faster than the tiles it fetches per call stream in over the link that feeds it, whether or
not the solve prices contention between transfers."""

from __future__ import annotations

from types import SimpleNamespace

from stream.opt.allocation.constraint_optimization.families.overlap import _core_stream_floor
from stream.opt.allocation.constraint_optimization.space import DecisionSpace


class _Route:
    targets = (SimpleNamespace(kind="memory"), SimpleNamespace(kind="compute"))


class _Quantities:
    def __contains__(self, name: str) -> bool:
        return False


def test_the_step_lasts_as_long_as_a_core_streams_its_tiles_in():
    """A 128x64 bf16 weight tile reaches its core in 2048 cycles over a 64-bit link, against 4096 for the firing's
    busiest link where the route crosses another column's tiles too: the step is bounded by the core's own share,
    after reuse."""
    added = []
    route = _Route()
    space = SimpleNamespace(
        core_stream_cycles=lambda tr, choice: 2048.0,
        transfer_latency_for_path=lambda tr, choice: 4096,
        route_name=lambda tr, choice: "w",
    )
    model = SimpleNamespace(add_constr=lambda constraint, name: added.append((constraint, name)))
    ctx = SimpleNamespace(space=space, quantities=_Quantities(), model=model)
    _core_stream_floor(ctx, overlap=0.0, iteration=3000.0, latency={("tr", route): 1024.0})
    assert added == [(True, "core_stream_w")]
    added.clear()
    _core_stream_floor(ctx, overlap=2600.0, iteration=3000.0, latency={("tr", route): 1024.0})
    assert added == [(False, "core_stream_w")]


def test_a_core_receives_its_own_share_of_a_firing_at_its_links_width():
    """Of the bits a firing delivers, the core receiving most takes them in over the links into it."""
    core, other = SimpleNamespace(kind="compute"), SimpleNamespace(kind="compute")
    links = {id(core): (SimpleNamespace(bandwidth=64),), id(other): (SimpleNamespace(bandwidth=64),)}
    path = SimpleNamespace(targets=(core, other), links_at=lambda memory, leaving: links[id(memory)])
    space = SimpleNamespace(moved_bits=lambda tr, choice, target_at: (131072.0, 65536.0)[target_at])
    assert DecisionSpace.core_stream_cycles(space, "tr", path) == 2048.0


def test_a_core_waits_for_the_weight_tile_its_memory_tile_streams_in(tmp_path):
    """One core calls a 32x128x64 kernel (1267 cycles, as measured) on a weight its memory tile holds across the rows
    and sends it a 128x64 tile per call: 131072 bits over the 64-bit link, 2048 cycles, set the step even where the
    solve prices no contention between transfers."""
    from pathlib import Path

    import yaml

    from stream.api import SolveOptions, default_families, evaluate_mapping
    from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
    from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload

    hardware = "stream/inputs/aie/hardware/whole_array_strix.yaml"
    library = {
        "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
        "kernel": {
            "matmul_bf16_bf16": {
                "family": "matmul",
                "dims": [{"name": "k", "divisor": 8}, {"name": "n", "divisor": 16}, {"name": "m", "divisor": 16}],
                "cycles": [{"m": 32, "k": 128, "n": 64, "cycles": 1267}],
            }
        },
    }
    workload = make_gemm_workload(256, 512, 64, "bf16", "bf16", str(tmp_path))
    mapping = make_gemm_mapping(256, 512, 64, 32, 128, 64, nb_rows_to_use=1, nb_cols_to_use=1, output_dir=str(tmp_path))
    tiled = yaml.safe_load(Path(mapping).read_text())
    tiled["fused_groups"][0]["intra_core_tiling"] = [{"dim": "Gemm.D1", "tile": 128}, {"dim": "Gemm.D0", "tile": 32}]
    Path(mapping).write_text(yaml.safe_dump(tiled, sort_keys=False))
    overlap = {"overlap": {"transfer_contention": False, "offchip_contention": False}}
    options = SolveOptions(
        nb_cols_to_use=1,
        kernel_library=library,
        families=default_families(hardware, options=overlap),
        artifacts=False,
    )
    latency = evaluate_mapping(hardware, workload, str(tmp_path), mapping, options).context.get("allocation")
    latency = latency.solution.latency
    assert latency.per_iteration - latency.overlap >= 128 * 64 * 16 / 64
