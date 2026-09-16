"""What a kernel library compiles, and what one call of it costs.

An accelerator's kernel sources decide which tile sizes are buildable and how fast each
one runs. Those are facts about that library, not about the mapper, so they arrive here
from the library itself and stream-dse holds no accelerator's numbers of its own. What it
holds is the two ways a source can declare its sizes -- a divisor, or an explicit list --
and a registry the library fills through :func:`adopt`.

An empty registry is the normal state for a mapper with no library attached: kernels keep
their declared shapes and the tile search simply has no block to offer.
"""

from math import log
from typing import Any

CALL_DIMS = ("m", "k", "n")

_LIBRARY: dict[str, dict[str, Any]] = {}


def adopt(manifest: dict[str, dict[str, Any]]) -> None:
    """Take this kernel library's declaration as the one in force."""
    _LIBRARY.clear()
    _LIBRARY.update(manifest)


def entry(symbol: str | None) -> dict[str, Any]:
    return _LIBRARY.get(symbol or "", {})


def blocks(symbol: str | None) -> dict[int, tuple[int, ...]]:
    """Sizes the library is built and timed at for ``symbol``, per call-dimension position.

    Only an explicit ``blocks`` list is an offer. A ``divisor`` is a legality rule -- what
    the source will accept -- and a rule is not a menu: every multiple of it compiles, but
    only the sizes somebody built and measured are worth choosing between. Reading the rule
    as an offer lets a search invent a block nobody ever ran. ``divisors`` carries the rule.
    """
    declared = entry(symbol)
    return {
        position: tuple(sizes)
        for position, name in enumerate(CALL_DIMS)
        if (sizes := declared.get("blocks", {}).get(name))
    }


def divisors(symbol: str | None) -> dict[int, int]:
    """The floor each call dimension must stay a multiple of, per position.

    A source declaring these is generic over them: it takes whatever block the rest of its
    group settles on, so it constrains that choice rather than proposing sizes of its own.
    """
    declared = entry(symbol).get("divisor", {})
    return {position: floor for position, name in enumerate(CALL_DIMS) if (floor := declared.get(name))}


def _ops(key: str) -> int:
    """Operations one call of this measured shape covers."""
    total = 1
    for part in key.split(","):
        total *= int(part)
    return total


def _nearest(measured: dict[str, Any], want: int) -> tuple[float, int] | None:
    """The measured call closest in size to ``want`` operations, as (cycles, operations).

    A kernel's efficiency is a property of the call it is compiled at, not of the family
    average: mm.cc measures 1595 cycles for a 64x64x64 call and 575 for a 32x32x64 one,
    which is 0.0061 against 0.0088 cycles per operation. Scaling every unmeasured shape
    from one per-operation anchor erases that, so a block nobody timed is priced as though
    it were as efficient as the largest one somebody did. Closest by size instead, which
    for a shape the same size as a measured call is that call.
    """
    sized = [(k, _ops(k)) for k in measured if k != "per_op"]
    if not sized or want <= 0:
        return None
    key, ops = min(sized, key=lambda kv: abs(log(kv[1] / want)))
    return measured[key], ops


def cycles(symbol: str | None, shape: dict[str, int]) -> float | tuple[float, int] | None:
    """Measured cycles for one call of this shape, or the closest measurement behind it."""
    measured = entry(symbol).get("cycles", {})
    key = ",".join(str(shape[name]) for name in CALL_DIMS if shape.get(name))
    if key in measured:
        return measured[key]
    want = 1
    for name in CALL_DIMS:
        want *= shape.get(name) or 1
    if near := _nearest(measured, want):
        return near
    if per_op := measured.get("per_op"):
        return per_op["cycles"], per_op["ops"]
    return None
