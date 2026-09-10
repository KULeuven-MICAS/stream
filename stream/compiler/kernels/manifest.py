"""What a kernel library compiles, and what one call of it costs.

An accelerator's kernel sources decide which tile sizes are buildable and how fast each
one runs. Those are facts about that library, not about the mapper, so they arrive here
from the library itself and stream-dse holds no accelerator's numbers of its own. What it
holds is the two ways a source can declare its sizes -- a divisor, or an explicit list --
and a registry the library fills through :func:`adopt`.

An empty registry is the normal state for a mapper with no library attached: kernels keep
their declared shapes and the tile search simply has no block to offer.
"""

from typing import Any

CALL_DIMS = ("m", "k", "n")

_LIBRARY: dict[str, dict[str, Any]] = {}


def adopt(manifest: dict[str, dict[str, Any]]) -> None:
    """Take this kernel library's declaration as the one in force."""
    _LIBRARY.clear()
    _LIBRARY.update(manifest)


def entry(symbol: str | None) -> dict[str, Any]:
    return _LIBRARY.get(symbol or "", {})


def _divisor_family(size: int, floor: int) -> tuple[int, ...]:
    """``size`` and every halving of it that stays a multiple of ``floor``, finest first."""
    out, block = [], size
    while block >= floor and block % floor == 0:
        out.append(block)
        block //= 2
    return tuple(reversed(out))


def blocks(symbol: str | None, shape: dict[str, int]) -> dict[int, tuple[int, ...]]:
    """Sizes the library compiles ``symbol`` at, per call-dimension position.

    ``shape`` is the kernel's own dimensions, which a divisor declaration is relative to:
    a source that takes any multiple of 16 still only offers what divides what it is
    being asked for.
    """
    declared = entry(symbol)
    if divisor := declared.get("divisor"):
        return {
            position: _divisor_family(shape[name], floor)
            for position, name in enumerate(CALL_DIMS)
            if (floor := divisor.get(name)) and shape.get(name)
        }
    return {
        position: tuple(sizes)
        for position, name in enumerate(CALL_DIMS)
        if (sizes := declared.get("blocks", {}).get(name))
    }


def cycles(symbol: str | None, shape: dict[str, int]) -> float | tuple[float, int] | None:
    """Measured cycles for one call of this shape, or the per-operation anchor behind it."""
    measured = entry(symbol).get("cycles", {})
    key = ",".join(str(shape[name]) for name in CALL_DIMS if shape.get(name))
    if key in measured:
        return measured[key]
    if per_op := measured.get("per_op"):
        return per_op["cycles"], per_op["ops"]
    return None
