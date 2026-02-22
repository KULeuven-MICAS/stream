from __future__ import annotations

from typing import Any

from stream.hardware.architecture.backends import AnyBackend, ZigZagCoreBackend


class Core:
    """A single hardware core in the Stream accelerator model.

    ``Core`` is a **thin identity object** with pluggable backend.  All
    hardware-specific details live inside a *backend* object that implements
    the backend protocol (``get_memory_capacity``, ``get_max_memory_bandwidth``,
    ``get_ir``).

    Access to backend attributes is transparent: ``core.operational_array``
    or ``core.mem_hierarchy_dict`` are resolved through ``__getattr__``
    delegation to the backend.
    """

    def __init__(
        self,
        *,
        core_id: int,
        name: str,
        core_type: str,
        backend: AnyBackend | None = None,
        utilization: int = 100,
        max_object_fifo_depth: int = 0,
        col_id: int | None = None,
        row_id: int | None = None,
    ):
        # ---- identity ----
        self.id: int = core_id
        self.name: str = name

        # ---- namespace / kind ----
        self.core_type: str = core_type
        self.type: str = self.core_type.split(".")[-1] if "." in self.core_type else self.core_type

        # ---- stream-specific attributes ----
        self.utilization: int = utilization
        self.max_object_fifo_depth: int = max_object_fifo_depth
        self.col_id: int | None = col_id
        self.row_id: int | None = row_id

        # ---- pluggable backend ----
        self._backend: AnyBackend | None = backend

    # ------------------------------------------------------------------ #
    # Backend access                                                     #
    # ------------------------------------------------------------------ #

    def to_zigzag_core(self) -> ZigZagCoreBackend:
        """Return the ZigZag backend.

        Use this when passing a core to ZigZag stages or cost models that
        expect a ``zigzag.hardware.architecture.accelerator.Accelerator``.

        Raises ``TypeError`` if the core has no ZigZag backend.
        """
        if not isinstance(self._backend, ZigZagCoreBackend):
            raise TypeError(f"{self} is not backed by a ZigZag core")
        return self._backend

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute look-ups to the backend."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        backend = self.__dict__.get("_backend")
        if backend is not None:
            try:
                return getattr(backend, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # ------------------------------------------------------------------ #
    # Namespace / kind helpers                                           #
    # ------------------------------------------------------------------ #

    @property
    def namespace(self) -> str:
        """The namespace prefix of ``core_type`` (e.g. ``'aie2'``, ``'zigzag'``)."""
        return self.core_type.split(".")[0] if "." in self.core_type else ""

    @property
    def kind(self) -> str:
        """The core kind suffix of ``core_type`` (e.g. ``'compute'``, ``'memory'``)."""
        return self.core_type.split(".")[-1] if "." in self.core_type else self.core_type

    # ------------------------------------------------------------------ #
    # Equality / hashing                                                 #
    # ------------------------------------------------------------------ #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Core):
            return NotImplemented
        if self.id != other.id:
            return False
        if self._backend is not None and other._backend is not None:
            if type(self._backend) is not type(other._backend):
                return False
            if isinstance(self._backend, ZigZagCoreBackend):
                return (
                    self._backend.operational_array == other._backend.operational_array
                    and self._backend.memory_hierarchy == other._backend.memory_hierarchy
                    and self._backend.dataflows == other._backend.dataflows
                )
            # For AIE2 and any future frozen-dataclass backends, __eq__ is auto-generated
            return self._backend == other._backend
        return True

    def has_same_performance(self, other: Core) -> bool:
        if self._backend is None or other._backend is None:
            return self.id == other.id
        if type(self._backend) is not type(other._backend):
            return False
        if isinstance(self._backend, ZigZagCoreBackend):
            return (
                self._backend.operational_array == other._backend.operational_array
                and self._backend.memory_hierarchy.has_same_performance(other._backend.memory_hierarchy)
                and self._backend.dataflows == other._backend.dataflows
            )
        return self._backend == other._backend

    def __hash__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return f"Core({self.id}, {self.core_type})"

    def __repr__(self) -> str:
        return str(self)

    # ------------------------------------------------------------------ #
    # Pickle support                                                     #
    # ------------------------------------------------------------------ #

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle, migrating old layouts if necessary."""
        if "_zigzag_core" in state and "_backend" not in state:
            state["_backend"] = state.pop("_zigzag_core")
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Stream-level memory interface  (delegates to backend)              #
    # ------------------------------------------------------------------ #

    def get_memory_capacity(self) -> int:
        """Total top-level memory capacity in bits."""
        assert self._backend is not None, f"{self} has no backend"
        return self._backend.get_memory_capacity()

    def get_max_memory_bandwidth(self, type: str) -> int:
        """Top-level memory read/write bandwidth in bits/cycle."""
        assert self._backend is not None, f"{self} has no backend"
        return self._backend.get_max_memory_bandwidth(type)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Serialization                                                      #
    # ------------------------------------------------------------------ #

    def _get_type_specific_ir(self) -> dict:
        """Return type-specific IR attributes based on the namespace."""
        if self.namespace == "aie2":
            return {"max_object_fifo_depth": self.max_object_fifo_depth}
        return {}

    def get_ir(self) -> dict:
        """Return a dictionary representation of this core for serialization."""
        d: dict = {
            "id": self.id,
            "name": self.name,
            "core_type": self.core_type,
            "type": self.type,
            "row_id": self.row_id,
            "col_id": self.col_id,
            "utilization": self.utilization,
        }

        # Merge backend-specific fields (uniform protocol)
        if self._backend is not None:
            d.update(self._backend.get_ir())

        # Merge type-specific attributes last
        d.update(self._get_type_specific_ir())
        return d
