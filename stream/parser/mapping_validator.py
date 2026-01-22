import logging
from typing import Any

from zigzag.parser.upgraded_validator import UpgradedValidator

logger = logging.getLogger(__name__)


class MappingValidator:
    """Class to validate user-given mappings from yaml file."""

    SCHEMA_LAYER: Any = {
        "name": {"type": "string", "required": True},
        "core_allocation": {
            "type": "list",
            "schema": {"type": "integer"},
            "required": True,
        },
        "inter_core_tiling": {
            "type": "list",
            "schema": {
                "type": "dict",
                "schema": {
                    "dim": {"type": "string", "required": True},
                    "split": {"type": "integer", "required": True},
                },
            },
            "required": False,
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

    SCHEMA_FUSED_GROUP: Any = {
        "name": {"type": "string", "required": True},
        "layers": {
            "type": "list",
            "schema": {"type": "string"},
            "required": True,
        },
        "intra_core_tiling": {
            "type": "list",
            "schema": {
                "type": "dict",
                "schema": {
                    "dim": {"type": "string", "required": True},
                    "tile": {"type": "integer", "required": True},
                },
            },
            "required": True,
            "default": [],
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object and normalize user-given data."""
        self.layer_validator = UpgradedValidator(is_array=True)
        self.fused_group_validator = UpgradedValidator(is_array=True)
        self.raw_data: Any = data
        self.normalized: dict[str, Any] = {"layers": [], "fused_groups": []}
        self.is_valid = True
        self.errors: list[str] = []

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Return normalized, user-provided data."""
        return self.normalized

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined mapping is invalid. %s", extra_msg)
        self.errors.append(extra_msg)

    def validate(self) -> bool:
        """Validate the user-provided mapping data."""
        root = self._coerce_root(self.raw_data)
        if not self.is_valid:
            return False

        self.normalized["layers"] = self.layer_validator.normalize_list(  # type: ignore[arg-type]
            root["layers"],
            schema=self.SCHEMA_LAYER,
        )

        for layer_data in self.normalized["layers"]:
            self._normalize_inter_core_tiling(layer_data)
            self.add_defaults(layer_data)

        if not self.layer_validator.validate(self.normalized["layers"], schema=self.SCHEMA_LAYER):  # type: ignore[arg-type]
            self.invalidate(f"The following layer restrictions apply: {self.layer_validator.errors}")

        fused_groups_raw = root.get("fused_groups", []) or []
        self.normalized["fused_groups"] = self.fused_group_validator.normalize_list(  # type: ignore[arg-type]
            fused_groups_raw,
            schema=self.SCHEMA_FUSED_GROUP,
        )
        for fused_group in self.normalized["fused_groups"]:
            self.add_fused_group_defaults(fused_group)

        if self.normalized["fused_groups"] and not self.fused_group_validator.validate(  # type: ignore[arg-type]
            self.normalized["fused_groups"],
            schema=self.SCHEMA_FUSED_GROUP,
        ):
            self.invalidate(f"The following fused group restrictions apply: {self.fused_group_validator.errors}")

        self._validate_fused_group_layer_references()
        self._validate_positive_tiling_values()

        return self.is_valid

    def _coerce_root(self, data: Any) -> dict[str, Any]:
        if isinstance(data, list):
            return {"layers": data, "fused_groups": []}
        if not isinstance(data, dict):
            self.invalidate("Mapping file must be a mapping with a top-level 'layers' list.")
            return {"layers": [], "fused_groups": []}
        if "layers" not in data:
            self.invalidate("Mapping file must contain a 'layers' entry.")
            return {"layers": [], "fused_groups": []}
        return data

    def _normalize_inter_core_tiling(self, layer_data: dict[str, Any]) -> None:
        raw_entries = layer_data.get("inter_core_tiling", []) or []
        normalized_entries: list[dict[str, Any]] = []
        for entry in raw_entries:
            if isinstance(entry, dict):
                normalized_entries.append(entry)
            else:
                self.invalidate(f"Invalid inter_core_tiling entry type: {type(entry)}")
        layer_data["inter_core_tiling"] = normalized_entries

    def add_defaults(self, layer_data: dict[str, Any]) -> None:
        kernel = layer_data.setdefault("kernel", {})
        kernel.setdefault("name", layer_data.get("name", ""))
        kernel.setdefault("kwargs", {"utilization": 100.0})
        layer_data.setdefault("inter_core_tiling", [])

    def add_fused_group_defaults(self, fused_group: dict[str, Any]) -> None:
        fused_group.setdefault("intra_core_tiling", [])

    def _validate_positive_tiling_values(self) -> None:
        for layer_data in self.normalized["layers"]:
            for entry in layer_data.get("inter_core_tiling", []):
                split_val = entry.get("split", 0)
                if not isinstance(split_val, int) or split_val <= 0:
                    self.invalidate(
                        f"Layer '{layer_data.get('name')}' split must be a positive integer; got {split_val}.",
                    )

        for fused_group in self.normalized["fused_groups"]:
            for entry in fused_group.get("intra_core_tiling", []) or []:
                tile_val = entry.get("tile", 0)
                if not isinstance(tile_val, int) or tile_val <= 0:
                    self.invalidate(
                        f"Fused group '{fused_group.get('name')}' tile must be a positive integer; got {tile_val}.",
                    )

    def _validate_fused_group_layer_references(self) -> None:
        layer_names = {layer.get("name") for layer in self.normalized.get("layers", []) if layer.get("name")}
        for fused_group in self.normalized["fused_groups"]:
            missing = [name for name in fused_group.get("layers", []) if name not in layer_names]
            if missing:
                self.invalidate(
                    f"Fused group '{fused_group.get('name', '<unknown>')}' references unknown layers {missing}.",
                )
