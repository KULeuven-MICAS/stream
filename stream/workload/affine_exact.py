"""Optional islpy-backed exact integer-set path for affine access queries, imported lazily so islpy stays optional and
:mod:`stream.workload.affine_access` works without it."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.workload.affine_access import map_dim_positions

Interval = tuple[int, int]

try:
    import islpy as _isl

    _ISL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the islpy-absent CI job
    _isl = None  # type: ignore[assignment]
    _ISL_AVAILABLE = False


def is_available() -> bool:
    """True when islpy is importable and the exact path can be used."""
    return _ISL_AVAILABLE


def _require() -> None:
    if not _ISL_AVAILABLE:
        raise RuntimeError("the exact affine path requires 'islpy'; install the 'stream[exact]' extra")


def _range_bounds(extent: range) -> Interval:
    if len(extent) == 0:
        raise ValueError("tile range must be non-empty")
    return extent[0], extent[-1]


def _box(tile: Mapping[int | AffineDimExpr, range]) -> dict[int, Interval]:
    box: dict[int, Interval] = {}
    for dim, extent in tile.items():
        position = dim.position if isinstance(dim, AffineDimExpr) else int(dim)
        box[position] = _range_bounds(extent)
    return box


def _domain_str(box: Mapping[int, Interval], num_dims: int) -> str:
    """ISL set over d0..d{n-1}: boxed dims to their interval, other dims fixed to 0."""
    variables = ", ".join(f"d{i}" for i in range(num_dims))
    constraints: list[str] = []
    for position in range(num_dims):
        if position in box:
            low, high = box[position]
            constraints.append(f"{low} <= d{position} <= {high}")
        else:
            constraints.append(f"d{position} = 0")
    body = " and ".join(constraints)
    return f"{{ [{variables}] : {body} }}" if body else f"{{ [{variables}] }}"


def _map_str(affine_map: AffineMap) -> str:
    variables = ", ".join(f"d{i}" for i in range(affine_map.num_dims))
    outputs = ", ".join(str(result) for result in affine_map.results)
    return f"{{ [{variables}] -> [{outputs}] }}"


def exact_footprint(affine_map: AffineMap, tile: Mapping[int | AffineDimExpr, range]) -> tuple[range, ...]:
    """Exact per-result index footprint of an iteration tile, via islpy."""
    _require()
    box = _box(tile)
    missing = set(map_dim_positions(affine_map)) - box.keys()
    if missing:
        raise ValueError(f"tile does not bound dimensions {sorted(missing)}")
    image = _isl.Map(_map_str(affine_map)).intersect_domain(_isl.Set(_domain_str(box, affine_map.num_dims))).range()
    ranges: list[range] = []
    for i in range(len(affine_map.results)):
        low = int(str(image.dim_min_val(i)))
        high = int(str(image.dim_max_val(i)))
        ranges.append(range(low, high + 1))
    return tuple(ranges)


def exact_compose_dependency(
    producer_out: AffineMap,
    consumer_in: AffineMap,
    consumer_tile: Mapping[int | AffineDimExpr, range],
    producer_domain: Mapping[int | AffineDimExpr, range] | None = None,
) -> dict[int, range]:
    """Exact per-dimension producer iteration region a consumer tile depends on; handles composite producer output maps
    the box path rejects, omits dims left unbounded, and accepts ``producer_domain`` to bound dims a composite output
    leaves coupled."""
    _require()
    consumer_box = _box(consumer_tile)
    missing = set(map_dim_positions(consumer_in)) - consumer_box.keys()
    if missing:
        raise ValueError(f"consumer tile does not bound dimensions {sorted(missing)}")
    consumed = (
        _isl.Map(_map_str(consumer_in))
        .intersect_domain(_isl.Set(_domain_str(consumer_box, consumer_in.num_dims)))
        .range()
    )
    region = _isl.Map(_map_str(producer_out)).intersect_range(consumed).domain()
    if producer_domain is not None:
        region = region.intersect(_isl.Set(_domain_str(_box(producer_domain), producer_out.num_dims)))
    result: dict[int, range] = {}
    for position in range(producer_out.num_dims):
        low_val = region.dim_min_val(position)
        high_val = region.dim_max_val(position)
        if low_val.is_infty() or low_val.is_neginfty() or high_val.is_infty() or high_val.is_neginfty():
            continue
        result[position] = range(int(str(low_val)), int(str(high_val)) + 1)
    return result
