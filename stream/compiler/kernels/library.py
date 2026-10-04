"""What a target's kernel library compiles, and what one call of it costs."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import log, prod
from pathlib import Path
from typing import Any

import yaml

FAMILIES = ("matmul", "vector")
_FAMILY_KEYS = {"ops_per_cycle", "mac"}
_KERNEL_KEYS = {"family", "binding", "object", "dims", "cycles", "per_op"}
_DIM_KEYS = {"name", "runtime", "fixed", "blocks", "divisor", "keep_whole"}


@dataclass(frozen=True)
class CallDim:
    name: str
    runtime: bool = False
    fixed: int | None = None
    blocks: tuple[int, ...] = ()
    divisor: int | None = None
    keep_whole: bool = False

    def accepts(self, size: int) -> bool:
        if self.fixed is not None and size != self.fixed:
            return False
        if self.blocks and size not in self.blocks:
            return False
        return not self.divisor or size % self.divisor == 0


@dataclass(frozen=True)
class Family:
    ops_per_cycle: float
    mac: Mapping[str, int] | None = None


@dataclass(frozen=True)
class KernelSpec:
    symbol: str
    family: str
    dims: tuple[CallDim, ...]
    binding: str | None = None
    object: str | None = None
    cycles: tuple[tuple[Mapping[str, int], float], ...] = ()
    per_op: tuple[float, int] | None = None

    def dim(self, name: str) -> CallDim | None:
        return next((d for d in self.dims if d.name == name), None)

    def validate(self, shape: Mapping[str, int]) -> None:
        for d in self.dims:
            if not d.accepts(shape[d.name]):
                raise ValueError(f"{self.symbol} does not compile {d.name}={shape[d.name]}")

    def call_cycles(self, shape: Mapping[str, int]) -> tuple[float, int] | None:
        """Cycles of the measured call nearest in size to ``shape``, and that call's operations."""
        for measured, cycles in self.cycles:
            if all(shape.get(name) == size for name, size in measured.items()):
                return cycles, prod(measured.values())
        want = prod(shape.values())
        if self.cycles and want > 0:
            measured, cycles = min(self.cycles, key=lambda mc: abs(log(prod(mc[0].values()) / want)))
            return cycles, prod(measured.values())
        return self.per_op


@dataclass(frozen=True)
class KernelLibrary:
    kernels: Mapping[str, KernelSpec]
    families: Mapping[str, Family] = field(default_factory=dict)

    def spec(self, symbol: str) -> KernelSpec | None:
        return self.kernels.get(symbol)

    @property
    def mac(self) -> Mapping[str, int]:
        """The matmul unit's tile, which is also the tiling an operand leaves a matmul in."""
        family = self.families.get("matmul")
        if family is None or family.mac is None:
            raise ValueError("the kernel library declares no matmul MAC tile")
        return family.mac

    @classmethod
    def load(cls, source: KernelLibrary | str | Path | Mapping[str, Any] | None) -> KernelLibrary | None:
        if source is None or isinstance(source, KernelLibrary):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            text = path.read_text()
            source = tomllib.loads(text) if path.suffix == ".toml" else yaml.safe_load(text)
        return cls.from_dict(source)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelLibrary:
        families = {name: _family(name, entry) for name, entry in data.get("family", {}).items()}
        kernels = {symbol: _kernel(symbol, entry, families) for symbol, entry in data.get("kernel", {}).items()}
        return cls(kernels=kernels, families=families)


def _family(name: str, entry: Mapping[str, Any]) -> Family:
    if name not in FAMILIES:
        raise ValueError(f"unknown kernel family {name!r}; the families are {FAMILIES}")
    if unknown := set(entry) - _FAMILY_KEYS:
        raise ValueError(f"family {name}: unknown keys {sorted(unknown)}")
    mac = entry.get("mac")
    return Family(float(entry["ops_per_cycle"]), {k: int(v) for k, v in mac.items()} if mac else None)


def _kernel(symbol: str, entry: Mapping[str, Any], families: Mapping[str, Family]) -> KernelSpec:
    if unknown := set(entry) - _KERNEL_KEYS:
        raise ValueError(f"{symbol}: unknown keys {sorted(unknown)}")
    if entry.get("family") not in families:
        raise ValueError(f"{symbol} names family {entry.get('family')!r}, which the library does not declare")
    dims = []
    for d in entry.get("dims", ()):
        if unknown := set(d) - _DIM_KEYS:
            raise ValueError(f"{symbol}: unknown dimension keys {sorted(unknown)}")
        dims.append(CallDim(**{**d, "blocks": tuple(d.get("blocks", ()))}))
    names = {d.name for d in dims}
    cycles = []
    for row in entry.get("cycles", ()):
        shape = {k: int(v) for k, v in row.items() if k != "cycles"}
        if unknown := set(shape) - names:
            raise ValueError(f"{symbol}: cycles name dimensions it does not declare: {sorted(unknown)}")
        cycles.append((shape, float(row["cycles"])))
    per_op = entry.get("per_op")
    return KernelSpec(
        symbol=symbol,
        family=entry["family"],
        dims=tuple(dims),
        binding=entry.get("binding"),
        object=entry.get("object"),
        cycles=tuple(cycles),
        per_op=(float(per_op["cycles"]), int(per_op["ops"])) if per_op else None,
    )
