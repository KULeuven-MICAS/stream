import logging
from typing import Any

from zigzag.parser.upgraded_validator import UpgradedValidator

logger = logging.getLogger(__name__)


class MappingValidator:
    """Class to validate user-given mappings from yaml file"""

    TILING_REGEX = r"^D[0-9]+, [0-9]+$"

    # Schema for a single operation, UpgradeValidator extrapolates to list of operations
    SCHEMA_SINGLE: Any = {
        "name": {"type": "string", "required": True},
        "core_allocation": {
            "type": "list",
            "schema": {"type": "integer"},
            "default": [0],
        },
        "inter_core_tiling": {
            "type": "list",
            "schema": {"type": "string", "regex": TILING_REGEX},
            "default": [],
        },
        "kernel": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": True},
                "kwargs": {"type": "dict", "required": True},
            },
            "required": False,
            "default": {},
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = UpgradedValidator(is_array=True)
        self.schema = MappingValidator.SCHEMA_SINGLE  # type: ignore
        self.data: list[dict[str, Any]] = self.validator.normalize_list(data, schema=self.schema)  # type: ignore
        self.is_valid = True
        self.errors = []

    @property
    def normalized_data(self):
        """! Return normalized, user-provided data."""
        # Can only be called after __init__, where data is automatically normalized
        return self.data

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined mapping is invalid. %s", extra_msg)
        self.errors.append(extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Add defaults where missing
        for mapping_data in self.data:
            self.add_defaults(mapping_data)

        # Validate according to schema
        validate_success = self.validator.validate(self.data, schema=self.schema)  # type: ignore
        errors = self.validator.errors
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        return self.is_valid

    def add_defaults(self, layer_data: dict[str, Any]) -> None:
        # Provide user-friendly defaults for missing kernel info
        kernel = layer_data.setdefault("kernel", {})
        kernel.setdefault("name", layer_data.get("name", ""))
        kernel.setdefault("kwargs", {"utilization": 100.0})
