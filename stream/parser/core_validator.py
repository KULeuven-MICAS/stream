import copy
import logging
from typing import Any

from zigzag.parser.accelerator_validator import AcceleratorValidator as ZigZagAcceleratorValidator

logger = logging.getLogger(__name__)


def core_kind_from_type(core_type: str | None) -> str:
    """Extracts the high-level kind (compute/memory) from a core type string."""
    if core_type is None:
        return ""
    if "." not in core_type:
        return core_type
    return core_type.split(".")[-1]


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
        """Normalize legacy type strings (e.g., `compute`) into a fully-qualified type."""
        if raw_type is None:
            return f"{default_namespace}.{default_kind}"
        if "." not in raw_type and raw_type in {"compute", "memory"}:
            return f"{default_namespace}.{raw_type}"
        return raw_type

    @classmethod
    def get_validator(cls, core_type: str) -> type["BaseCoreValidator"] | None:
        return cls._registry.get(core_type)

    @classmethod
    def supported_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


class BaseCoreValidator(ZigZagAcceleratorValidator):
    """Base class for all core validators with common schema and error collection."""

    CORE_NAMESPACE = CoreValidatorRegistry.default_namespace
    CORE_KIND = "compute"

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
        schema = copy.deepcopy(ZigZagAcceleratorValidator.SCHEMA)
        schema.update(
            {
                "type": {"type": "string", "required": False, "default": cls.core_type()},
                "utilization": {"type": "float", "required": False},
            }
        )
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


@CoreValidatorRegistry.register
class ZigZagComputeCoreValidator(BaseCoreValidator):
    CORE_KIND = "compute"
    CORE_NAMESPACE = "zigzag"


@CoreValidatorRegistry.register
class ZigZagMemoryCoreValidator(BaseCoreValidator):
    CORE_KIND = "memory"
    CORE_NAMESPACE = "zigzag"


@CoreValidatorRegistry.register
class AIE2ComputeCoreValidator(BaseCoreValidator):
    CORE_KIND = "compute"
    CORE_NAMESPACE = "aie2"


@CoreValidatorRegistry.register
class AIE2MemoryCoreValidator(BaseCoreValidator):
    CORE_KIND = "memory"
    CORE_NAMESPACE = "aie2"
