import copy
import logging
from typing import Any

from cerberus import Validator
from zigzag.parser.accelerator_validator import AcceleratorValidator as ZigZagAcceleratorValidator

logger = logging.getLogger(__name__)


# =============================================================================
# Allowed namespaces and kinds
# =============================================================================

ALLOWED_NAMESPACES: frozenset[str] = frozenset({"zigzag", "aie2"})
"""Namespace prefixes recognised by the core type system (e.g. ``"zigzag"``, ``"aie2"``)."""

ALLOWED_KINDS: frozenset[str] = frozenset({"compute", "memory", "shim", "offchip"})
"""Core kind suffixes recognised by the core type system.

* ``compute`` — processing tile (systolic array, SIMD, …)
* ``memory``  — on-chip memory / cache tile
* ``shim``    — interface / DMA tile (e.g. AIE2 shim-DMA)
* ``offchip`` — off-chip DRAM or external memory controller
"""


def core_kind_from_type(core_type: str | None) -> str:
    """Extract the kind suffix from a fully-qualified core type string.

    >>> core_kind_from_type("aie2.compute")
    'compute'
    >>> core_kind_from_type("memory")
    'memory'
    >>> core_kind_from_type(None)
    ''
    """
    if core_type is None:
        return ""
    if "." not in core_type:
        return core_type
    return core_type.split(".")[-1]


# =============================================================================
# Per-namespace / per-kind core schema extensions
# -----------------------------------------------------------------------------
# _BASE_CORE_SCHEMA       — fields every core YAML may carry (type string).
# _<NS>_EXTRA_SCHEMA      — namespace-level additions (e.g. max_object_fifo_depth
#                            for aie2).
# _<NS>_<KIND>_EXTRA_SCHEMA — kind-specific overrides within a namespace.
#
# HOW TO ADD A NEW NAMESPACE
# --------------------------
#   1. Add the namespace string to ALLOWED_NAMESPACES.
#   2. Define _<NS>_EXTRA_SCHEMA here with namespace-wide fields.
#   3. (Optional) Define _<NS>_<KIND>_EXTRA_SCHEMA for kind-specific fields.
#   4. Create a <NS>BaseCoreValidator that sets EXTRA_SCHEMA.
#   5. For each kind, create a leaf class with CORE_KIND and register it.
# =============================================================================

_BASE_CORE_SCHEMA: dict[str, Any] = {
    # Fully-qualified core type string (e.g. "aie2.compute").  Defaults to
    # "<namespace>.<kind>" inferred from the validator class used.
    "type": {"type": "string", "required": False},
}

# ZigZag cores need nothing beyond the base schema.
_ZIGZAG_EXTRA_SCHEMA: dict[str, Any] = {}

# AIE2 cores carry the object-FIFO depth as a tile-level hardware property.
# Each tile type specifies its own limit so mixed-depth arrays are supported.
_AIE2_EXTRA_SCHEMA: dict[str, Any] = {
    "max_object_fifo_depth": {
        "type": "integer",
        "required": True,
        "min": 1,
    },
}

# AIE2 *compute* tiles additionally carry a utilization percentage that
# captures the fraction of peak throughput achievable on this tile.
_AIE2_COMPUTE_EXTRA_SCHEMA: dict[str, Any] = {
    **_AIE2_EXTRA_SCHEMA,
    "utilization": {
        "type": "float",
        "required": False,
        "default": 100,
    },
}


class CoreValidatorRegistry:
    """Registry that maps a core type string to its validator class."""

    default_namespace = "zigzag"
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, validator_cls: type):
        cls._registry[validator_cls.core_type()] = validator_cls
        return validator_cls

    @classmethod
    def normalize_core_type(cls, raw_type: str | None, *, default_namespace: str, default_kind: str) -> str:
        """Normalize legacy type strings (e.g., ``"compute"``) into a fully-qualified ``"<ns>.<kind>"``."""
        if raw_type is None:
            return f"{default_namespace}.{default_kind}"
        if "." not in raw_type and raw_type in ALLOWED_KINDS:
            return f"{default_namespace}.{raw_type}"
        return raw_type

    @classmethod
    def get_validator(cls, core_type: str) -> type | None:
        return cls._registry.get(core_type)

    @classmethod
    def supported_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# =============================================================================
# ZigZag namespace validators  (inherit from ZigZagAcceleratorValidator)
# =============================================================================


class ZigZagBaseCoreValidator(ZigZagAcceleratorValidator):
    """Base class for all ``zigzag.*`` core validators.

    These validators inherit from ``ZigZagAcceleratorValidator`` and expect
    the full ZigZag YAML format (``memories``, ``operational_array``, …).

    For each core kind, subclass this, set ``CORE_KIND``, and decorate
    with ``@CoreValidatorRegistry.register``.
    """

    CORE_NAMESPACE: str = "zigzag"
    CORE_KIND: str = "compute"
    EXTRA_SCHEMA: dict[str, Any] = _ZIGZAG_EXTRA_SCHEMA

    def __init__(self, data: Any):
        self.errors: list[str] = []
        self.core_type = CoreValidatorRegistry.normalize_core_type(
            data.get("type") if isinstance(data, dict) else None,
            default_namespace=self.CORE_NAMESPACE,
            default_kind=self.CORE_KIND,
        )
        super().__init__(data)
        self.validator.schema = self._build_schema()
        self.data = self.validator.normalized(data)  # type: ignore
        self.is_valid = True

    @classmethod
    def core_type(cls) -> str:
        return f"{cls.CORE_NAMESPACE}.{cls.CORE_KIND}"

    @classmethod
    def _build_schema(cls) -> dict[str, Any]:
        """Compose the full schema: ZigZag base + universal Stream fields + namespace extras."""
        schema = copy.deepcopy(ZigZagAcceleratorValidator.SCHEMA)
        base = copy.deepcopy(_BASE_CORE_SCHEMA)
        base["type"]["default"] = cls.core_type()
        schema.update(base)
        schema.update(copy.deepcopy(cls.EXTRA_SCHEMA))
        return schema

    def invalidate(self, extra_msg: str):
        self.errors.append(extra_msg)
        self.is_valid = False
        logger.critical("User-defined core is invalid. %s", extra_msg)

    def validate(self) -> bool:
        self.data["type"] = CoreValidatorRegistry.normalize_core_type(
            self.data.get("type"),
            default_namespace=self.CORE_NAMESPACE,
            default_kind=self.CORE_KIND,
        )

        if not CoreValidatorRegistry.get_validator(self.data["type"]):
            supported = ", ".join(CoreValidatorRegistry.supported_types())
            self.invalidate(f"Unsupported core type '{self.data['type']}'. Supported types: {supported}")

        if core_kind_from_type(self.data["type"]) != self.CORE_KIND:
            self.invalidate(
                f"Core type '{self.data['type']}' must map to kind '{self.CORE_KIND}' "
                f"(found '{core_kind_from_type(self.data['type'])}')."
            )

        parent_valid = super().validate()
        self.is_valid = self.is_valid and parent_valid
        return self.is_valid

    @property
    def normalized_data(self) -> dict[str, Any]:
        return self.data


@CoreValidatorRegistry.register
class ZigZagComputeCoreValidator(ZigZagBaseCoreValidator):
    CORE_KIND = "compute"


@CoreValidatorRegistry.register
class ZigZagMemoryCoreValidator(ZigZagBaseCoreValidator):
    CORE_KIND = "memory"


@CoreValidatorRegistry.register
class ZigZagShimCoreValidator(ZigZagBaseCoreValidator):
    CORE_KIND = "shim"


@CoreValidatorRegistry.register
class ZigZagOffchipCoreValidator(ZigZagBaseCoreValidator):
    CORE_KIND = "offchip"


# =============================================================================
# AIE2 namespace validators  (standalone cerberus — no ZigZag inheritance)
# =============================================================================
# AIE2 core YAMLs use a simplified format:
#     name, type, max_object_fifo_depth, memory.capacity
# They do NOT carry ZigZag memories / operational_array / dataflows.

_AIE2_NATIVE_SCHEMA: dict[str, Any] = {
    "name": {"type": "string", "required": True},
    "type": {"type": "string", "required": False},
    "max_object_fifo_depth": {
        "type": "integer",
        "required": True,
        "min": 1,
    },
    "memory": {
        "type": "dict",
        "required": True,
        "schema": {
            "capacity": {"type": "integer", "required": True, "min": 0},
            "bandwidth_min": {"type": "integer", "required": True, "min": 0},
            "bandwidth_max": {"type": "integer", "required": True, "min": 0},
        },
    },
}


class AIE2BaseCoreValidator:
    """Standalone cerberus-based validator for ``aie2.*`` cores.

    Unlike the ZigZag validators, this does **not** inherit from
    ``ZigZagAcceleratorValidator``.  AIE2 core YAMLs use a simplified
    schema that only carries ``name``, ``type``, ``max_object_fifo_depth``,
    and ``memory.capacity``.
    """

    CORE_NAMESPACE: str = "aie2"
    CORE_KIND: str = "compute"
    #: Cerberus schema for this validator (may be overridden by subclasses).
    SCHEMA: dict[str, Any] = _AIE2_NATIVE_SCHEMA

    def __init__(self, data: Any):
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}
        self.errors: list[str] = []
        self.is_valid: bool = True

        # Build a cerberus Validator with the (possibly extended) schema
        schema = copy.deepcopy(self.SCHEMA)
        schema.setdefault("type", {})
        schema["type"]["default"] = self.core_type()
        self._validator = Validator(schema, allow_unknown=False)

    @classmethod
    def core_type(cls) -> str:
        return f"{cls.CORE_NAMESPACE}.{cls.CORE_KIND}"

    def invalidate(self, extra_msg: str):
        self.errors.append(extra_msg)
        self.is_valid = False
        logger.critical("User-defined core is invalid. %s", extra_msg)

    def validate(self) -> bool:
        # Normalize the type field
        self.data["type"] = CoreValidatorRegistry.normalize_core_type(
            self.data.get("type"),
            default_namespace=self.CORE_NAMESPACE,
            default_kind=self.CORE_KIND,
        )

        if not CoreValidatorRegistry.get_validator(self.data["type"]):
            supported = ", ".join(CoreValidatorRegistry.supported_types())
            self.invalidate(f"Unsupported core type '{self.data['type']}'. Supported types: {supported}")

        if core_kind_from_type(self.data["type"]) != self.CORE_KIND:
            self.invalidate(
                f"Core type '{self.data['type']}' must map to kind '{self.CORE_KIND}' "
                f"(found '{core_kind_from_type(self.data['type'])}')."
            )

        if not self._validator.validate(self.data):
            for field, msgs in self._validator.errors.items():
                self.invalidate(f"Field '{field}': {msgs}")

        # Apply cerberus defaults / coercions
        self.data = self._validator.document or self.data
        return self.is_valid

    @property
    def normalized_data(self) -> dict[str, Any]:
        return self.data


@CoreValidatorRegistry.register
class AIE2ComputeCoreValidator(AIE2BaseCoreValidator):
    """AIE2 compute tile.

    In addition to the base AIE2 fields (``max_object_fifo_depth``), compute
    tiles carry an optional ``utilization`` percentage (0–100 %) that captures
    the fraction of peak throughput achievable on this tile.
    """

    CORE_KIND = "compute"
    SCHEMA: dict[str, Any] = {
        **_AIE2_NATIVE_SCHEMA,
        "utilization": {
            "type": "float",
            "required": False,
            "default": 100,
        },
    }


@CoreValidatorRegistry.register
class AIE2MemoryCoreValidator(AIE2BaseCoreValidator):
    CORE_KIND = "memory"


@CoreValidatorRegistry.register
class AIE2ShimCoreValidator(AIE2BaseCoreValidator):
    """AIE2 shim / DMA interface tile (row 0 of the AIE array)."""

    CORE_KIND = "shim"


@CoreValidatorRegistry.register
class AIE2OffchipCoreValidator(AIE2BaseCoreValidator):
    CORE_KIND = "offchip"
