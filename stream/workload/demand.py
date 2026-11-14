"""What a transfer's targets read of its tensor, as boxes of the tensor, and the bits a set of boxes covers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

Box = tuple[tuple[int, int], ...]
"""Inclusive index range per axis of the data a demand moves; equal boxes are the same data."""


@dataclass(frozen=True)
class DemandItem:
    """Data the target at ``target`` reads, ``box`` of the tensor, held by each source at ``sources``, any one of which
    can serve it; or, if ``reduce``, of which each holds a partial sum that the target needs all of. ``copies`` are the
    transfer's copies that read it there; with no ``sources``, it is a local copy from the larger copy ``copied_from``
    in the target's memory. With ``firings`` above 1, the first axis of ``box`` is the range of those firings the item
    stands for."""

    target: int
    sources: tuple[int, ...]
    box: Box
    reduce: bool = False
    copies: tuple[int, ...] = ()
    copied_from: int | None = None
    firings: int = 1


def firing_count(items: Iterable[DemandItem]) -> int:
    """The firings a demand's boxes span, 1 where one firing stands for all."""
    return max((item.firings for item in items), default=1)


def copy_sets(items: Iterable[DemandItem]) -> tuple[tuple[int, ...], ...]:
    """The copy sets ``items`` deliver some box to: each a copy alone, or several reading the same box."""
    return tuple(dict.fromkeys(item.copies for item in items))


def moved_elements(items: Iterable[DemandItem]) -> float:
    """Points the delivered ``items`` carry per firing: what each source sends counted once however many targets read
    it, and a partial sum once whichever sources hold it."""
    items = list(items)
    sent: dict[int | None, list[Box]] = {}
    for item in items:
        sent.setdefault(None if item.reduce else item.sources[0], []).append(item.box)
    return sum(covered(boxes) for boxes in sent.values()) / firing_count(items)


def item_bits(items: Iterable[DemandItem], bitwidth: float) -> float:
    """Bits per firing the boxes of ``items`` cover together, distinct data counted once."""
    items = list(items)
    return box_bits((item.box for item in items), bitwidth) / firing_count(items)


def box_bits(boxes: Iterable[Box], bitwidth: float) -> float:
    """Bits the ``boxes`` cover together: distinct data counted once."""
    return covered(boxes) * bitwidth


def covered(boxes: Iterable[Box]) -> int:
    """Points the inclusive boxes cover together."""
    return _covered(list(dict.fromkeys(boxes)))


def _covered(boxes: list[Box]) -> int:
    """Points the inclusive boxes cover together, sweeping the first axis between the boxes' edges."""
    if not boxes:
        return 0
    if len(boxes) == 1:
        return sum(prod(hi - lo + 1 for lo, hi in box) for box in boxes)
    if len(boxes[0]) == 1:
        spans = sorted((box[0][0], box[0][1]) for box in boxes)
        total, end = 0, -1
        for lo, hi in spans:
            if hi > end:
                total += hi - max(lo, end + 1) + 1
                end = hi
        return total
    edges = sorted({box[0][0] for box in boxes} | {box[0][1] + 1 for box in boxes})
    total = 0
    for lo, nxt in zip(edges, edges[1:], strict=False):
        inside = [box[1:] for box in boxes if box[0][0] <= lo and box[0][1] >= nxt - 1]
        total += (nxt - lo) * _covered(list(dict.fromkeys(inside)))
    return total


def _disjoint(a: Box, b: Box) -> bool:
    return any(ha < lb or hb < la for (la, ha), (lb, hb) in zip(a, b, strict=True))


def meet(a: Box, b: Box) -> Box | None:
    """The part two boxes share, if any."""
    if _disjoint(a, b):
        return None
    return tuple((max(la, lb), min(ha, hb)) for (la, ha), (lb, hb) in zip(a, b, strict=True))


def within(inner: Box, outer: Box) -> bool:
    """Whether ``outer`` holds all of ``inner``."""
    return all(lo <= ilo and ihi <= hi for (ilo, ihi), (lo, hi) in zip(inner, outer, strict=True))


def less(box: Box, cuts: Iterable[Box]) -> list[Box]:
    """``box`` less all of ``cuts``, as disjoint boxes."""
    rest = [box]
    for cut in cuts:
        rest = [part for piece in rest for part in minus(piece, cut)]
    return rest


def minus(box: Box, cut: Box) -> list[Box]:
    """``box`` less ``cut``, as disjoint boxes: per axis the slabs of ``box`` below and above ``cut``."""
    if meet(box, cut) is None:
        return [box]
    rest, parts = list(box), []
    for axis, ((lo, hi), (cut_lo, cut_hi)) in enumerate(zip(box, cut, strict=True)):
        if lo < cut_lo:
            parts.append((*rest[:axis], (lo, cut_lo - 1), *rest[axis + 1 :]))
        if cut_hi < hi:
            parts.append((*rest[:axis], (cut_hi + 1, hi), *rest[axis + 1 :]))
        rest[axis] = (max(lo, cut_lo), min(hi, cut_hi))
    return parts
