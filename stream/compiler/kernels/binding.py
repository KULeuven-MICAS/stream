"""Kernel bindings: the symbol, object and signature whoever provides a kernel declares for it.

A binding is any object with ``name``, ``object_file_name`` and ``arg_types()`` (numpy array or
scalar types), such as mlir-aie's ``ExternalFunction``; a call made beside it is an attribute of it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from math import prod
from pathlib import Path
from typing import Any, Protocol, get_args

import numpy as np
from xdsl.dialects.builtin import MemRefType, bf16, f32, i8, i16, i32
from xdsl.ir import Attribute

ELEMENT_TYPES: dict[str, Attribute] = {"bfloat16": bf16, "float32": f32, "int8": i8, "int16": i16, "int32": i32}


class Binding(Protocol):
    name: str
    object_file_name: str

    def arg_types(self) -> list | None: ...


@dataclass(frozen=True)
class Declared:
    """Stream's own declaration of a kernel no provider binds, linked from the library's object."""

    name: str
    object_file_name: str

    def arg_types(self) -> None:
        return None


def resolve(entry: dict[str, Any]) -> Binding:
    """The binding a ``kernels.json`` entry names: a provider's, or stream's own declaration."""
    if "binding" not in entry:
        return Declared(entry["symbol"], entry["object"])
    module, function = entry["binding"].split(":")
    binding = getattr(import_module(module), function)(**entry["args"])
    return getattr(binding, entry["companion"]) if "companion" in entry else binding


@dataclass
class Bindings:
    """The bindings one design's calls resolve to, for the ``npu`` it targets, keyed by their entries."""

    npu: str
    used: dict[str, Binding] = field(default_factory=dict)

    def resolve(self, entry: dict[str, Any]) -> Binding:
        key = json.dumps(entry, sort_keys=True)
        if key not in self.used:
            self.used[key] = resolve(entry)
        return self.used[key]

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([json.loads(key) for key in self.used], indent=2) + "\n")


def load_bindings(path: str | Path) -> list[Binding]:
    """One binding per object the design links, rebuilt from the ``kernels.json`` codegen wrote beside it."""
    objects: dict[str, Binding] = {}
    for entry in json.loads(Path(path).read_text()):
        binding = resolve(entry)
        objects.setdefault(binding.object_file_name, binding)
    return list(objects.values())


def _accepts(expected: Any, actual: Attribute) -> bool:
    if not (array := get_args(expected)):
        return actual == ELEMENT_TYPES.get(np.dtype(expected).name)
    shape, dtype = array
    return (
        isinstance(actual, MemRefType)
        and prod(actual.get_shape()) == prod(shape)
        and actual.element_type == ELEMENT_TYPES.get(np.dtype(get_args(dtype)[0]).name)
    )


def check(call: str, binding: Binding, operands: Sequence[Attribute]) -> None:
    """Raise unless stream's call to ``call`` passes what ``binding`` declares; stream's own declarations pass."""
    if (expected := binding.arg_types()) is None:
        return
    if len(expected) != len(operands):
        raise ValueError(
            f"kernel {call} ({binding.name}) takes {len(expected)} arguments, but stream passes {len(operands)}"
        )
    for index, (declared, passed) in enumerate(zip(expected, operands, strict=True)):
        if not _accepts(declared, passed):
            raise ValueError(f"kernel {call} ({binding.name}) takes {declared} as argument {index}, not {passed}")
