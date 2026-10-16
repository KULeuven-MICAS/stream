"""How a tensor copy is laid out in memory, and what moving it into another layout costs.

A layout orders a tensor's axes from outermost to innermost, the innermost contiguous. Each copy of a tensor has one:
a model parameter is packed offline in whatever order the kernel reading it needs, a host input and output are
row-major, a node writes its output with the axes its array produces together innermost, and a copy a kernel reads
has the axes the kernel reads together innermost (``contiguous axes``). A transfer moving a tile between two layouts
converts it on the fly, as a DMA does: the data streams in one layout's order, so one side accesses it contiguously and
the other in runs only as long as the innermost axes the two layouts share. How long a memory's contiguous runs are
sets how much of its bandwidth a transfer gets (``BandwidthModel.efficiency``), which is all a layout costs here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """The axes of a tensor from outermost to innermost."""

    order: tuple[int, ...]

    @classmethod
    def row_major(cls, rank: int) -> Layout:
        return cls(tuple(range(rank)))

    def with_innermost(self, axes: Iterable[int]) -> Layout:
        """This layout with ``axes`` moved innermost, every axis keeping its order relative to the others it moves
        with; itself when they already are innermost, in any order."""
        inner = set(axes)
        if not inner or set(self.order[len(self.order) - len(inner) :]) == inner:
            return self
        return Layout(tuple(a for a in self.order if a not in inner) + tuple(a for a in self.order if a in inner))

    def data_order(self, full: tuple[int, ...]) -> tuple[int, ...]:
        """The order of the axes of a ``full`` tensor that hold more than one element: an axis of one element places
        its data nowhere, so two layouts differing only in where it sits hold the same bytes in the same order."""
        return tuple(axis for axis in self.order if full[axis] != 1)

    def same_data_order(self, other: Layout, full: tuple[int, ...]) -> bool:
        return self.data_order(full) == other.data_order(full)

    def strides(self, extent: tuple[int, ...]) -> tuple[int, ...]:
        """Elements between neighbours along each axis of a buffer of ``extent`` laid out so."""
        strides, step = [0] * len(extent), 1
        for axis in reversed(self.order):
            strides[axis], step = step, step * extent[axis]
        return tuple(strides)

    def access(self, block: tuple[int, ...], extent: tuple[int, ...], walk: Layout) -> list[tuple[int, int]]:
        """The (size, stride) loops, outermost first, in elements, that visit ``block`` of a buffer of ``extent`` laid
        out so in ``walk``'s axis order: loops of one element dropped and a loop continuing the one inside it merged
        into it, as a DMA descriptor writes them."""
        strides = self.strides(extent)
        loops: list[tuple[int, int]] = []
        for axis in reversed(walk.order):
            size, stride = block[axis], strides[axis]
            if size == 1:
                continue
            if loops and stride == loops[0][0] * loops[0][1]:
                loops[0] = (loops[0][0] * size, loops[0][1])
            else:
                loops.insert(0, (size, stride))
        return loops or [(1, 1)]

    def contiguous_bytes(self, block: tuple[int, ...], full: tuple[int, ...], element_bits: int) -> float:
        """Bytes a ``block`` of a ``full`` tensor laid out so covers in one contiguous run: its innermost axes up to
        and including the first it does not span whole."""
        span = 1
        for axis in reversed(self.data_order(full)):
            span *= block[axis]
            if block[axis] != full[axis]:
                break
        return span * element_bits / 8


def shared_run_bytes(  # noqa: PLR0913
    block: tuple[int, ...],
    full: tuple[int, ...],
    element_bits: int,
    one: Layout,
    other: Layout,
    extents: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> float:
    """Bytes of ``block`` contiguous in both layouts at once: the innermost axes they order alike, up to and including
    the first the block does not span whole in either buffer (``extents``, the whole tensor where not given); one
    element where their innermost axes differ."""
    first, second = extents or (full, full)
    span = 1
    for axis, same in zip(reversed(one.data_order(full)), reversed(other.data_order(full)), strict=True):
        if axis != same:
            break
        span *= block[axis]
        if block[axis] != first[axis] or block[axis] != second[axis]:
            break
    return span * element_bits / 8


Rate = Callable[[float], float]
"""Bits per cycle one side of a transfer moves at contiguous runs of this many bytes, infinite where nothing bounds
it."""


def transfer_runs(*args, **kwargs) -> tuple[float, float]:
    """The contiguous bytes per run a transfer reads and writes (see :func:`transfer_stream`)."""
    _, read, write = transfer_stream(*args, **kwargs)
    return read, write


def transfer_stream(  # noqa: PLR0913
    block: tuple[int, ...],
    full: tuple[int, ...],
    element_bits: int,
    source: Layout,
    target: Layout,
    read_rate: Rate,
    write_rate: Rate,
    extents: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> tuple[Layout, float, float]:
    """The layout a transfer streams ``block`` of a ``full`` tensor in, and the contiguous bytes per run it reads and
    writes, from a ``source`` layout to a ``target`` one, each in the buffer that side holds (``extents``: the
    source's and the target's, the whole tensor where not given). Between two layouts the DMA gathers in the target's
    order, reading runs only as long as the layouts share, or scatters in the source's order, writing such runs,
    whichever the slower of its two sides moves faster."""
    source_extent, target_extent = extents or (full, full)
    read = source.contiguous_bytes(block, source_extent, element_bits)
    write = target.contiguous_bytes(block, target_extent, element_bits)
    if source.same_data_order(target, full):
        return source, read, write
    shared = shared_run_bytes(block, full, element_bits, source, target, (source_extent, target_extent))
    gather, scatter = (target, shared, write), (source, read, shared)
    return max(gather, scatter, key=lambda runs: min(read_rate(runs[1]), write_rate(runs[2])))
