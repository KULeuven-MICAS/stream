import copy
import logging
from typing import Any

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
    _registry: dict[str, type["BaseCoreValidator"]] = {}

    @classmethod
    def register(cls, validator_cls: type["BaseCoreValidator"]):
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
    def get_validator(cls, core_type: str) -> type["BaseCoreValidator"] | None:
        return cls._registry.get(core_type)

    @classmethod
    def supported_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


class BaseCoreValidator(ZigZagAcceleratorValidator):
    """Base class for all core validators.

    Subclasses should not be registered directly.  Instead:
      - Create a namespace-specific intermediate class (e.g.
        ``AIE2BaseCoreValidator``) that sets ``EXTRA_SCHEMA``.
      - For each core kind in that namespace, subclass the intermediate class,
        set ``CORE_KIND``, and decorate with ``@CoreValidatorRegistry.register``.
    """

    CORE_NAMESPACE: str = CoreValidatorRegistry.default_namespace
    CORE_KIND: str = "compute"
    #: Per-namespace schema additions — see module-level _*_EXTRA_SCHEMA dicts.
    EXTRA_SCHEMA: dict[str, Any] = {}

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
        # Deep-copy the module-level constants before mutating so repeated calls
        # across different cls values never clobber the originals.
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
        # Normalize and verify the core type before running the ZigZag validation
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
        self.post_validate()
        return self.is_valid

    def post_validate(self) -> None:
        """Optional hook for subclasses to add extra validation."""
        return

    @property
    def normalized_data(self) -> dict[str, Any]:
        return self.data


# =============================================================================
# ZigZag namespace cores
# =============================================================================


class ZigZagBaseCoreValidator(BaseCoreValidator):
    """Shared base for all ``zigzag.*`` cores.  No extra fields required."""

    CORE_NAMESPACE = "zigzag"
    EXTRA_SCHEMA = _ZIGZAG_EXTRA_SCHEMA


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
# AIE2 namespace cores
# =============================================================================


class AIE2BaseCoreValidator(BaseCoreValidator):
    """Shared base for all ``aie2.*`` cores.

    Extra required field: ``max_object_fifo_depth`` (positive integer) —
    the maximum number of object-FIFO slots available in this tile's
    memory.  Defined at core level so each tile type can carry its own
    hardware-imposed limit, allowing mixed-depth arrays without any
    accelerator-level dispatch table.

    Individual kinds may add further fields by overriding
    :attr:`EXTRA_SCHEMA` (see :class:`AIE2ComputeCoreValidator`).
    """

    CORE_NAMESPACE = "aie2"
    EXTRA_SCHEMA = _AIE2_EXTRA_SCHEMA


@CoreValidatorRegistry.register
class AIE2ComputeCoreValidator(AIE2BaseCoreValidator):
    """AIE2 compute tile.

    In addition to the base AIE2 fields (``max_object_fifo_depth``), compute
    tiles carry an optional ``utilization`` percentage (0–100 %) that captures
    the fraction of peak throughput achievable on this tile.
    """

    CORE_KIND = "compute"
    EXTRA_SCHEMA = _AIE2_COMPUTE_EXTRA_SCHEMA


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
