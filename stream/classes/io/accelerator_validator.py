import logging
import os
from typing import Any
from cerberus import Validator

from zigzag.parser.AcceleratorValidator import AcceleratorValidator as CoreValidator
from zigzag.utils import open_yaml

logger = logging.getLogger(__name__)


class AcceleratorValidator:
    INPUT_DIR_LOCATION = "stream/inputs/"

    SCHEMA = {
        "name": {"type": "string", "required": True},
        "cores": {
            "type": "list",
            "required": True,
            "schema": {
                "type": "string",
                "regex": r"^(?:[a-zA-Z0-9_\-\.]+|(\.\/|\.\.\/)[a-zA-Z0-9_\-\.\/]*)$",
            },
        },
        "graph": {
            "type": "dict",
            "required": True,
            "schema": {
                "type": {"type": "string", "required": True},
                "nb_rows": {"type": "integer", "required": False},
                "nb_cols": {"type": "integer", "required": False},
                "bandwidth": {"type": "integer", "required": True},
                "unit_energy_cost": {"type": "float", "default": 0},
                "pooling_core_id": {
                    "type": "integer",
                    "nullable": True,
                    "default": None,
                },
                "simd_core_id": {
                    "type": "integer",
                    "nullable": True,
                    "default": None,
                },
                "offchip_core_id": {
                    "type": "integer",
                    "nullable": True,
                    "default": None,
                },
            },
        },
    }

    def __init__(self, data: Any, accelerator_path: str):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = Validator()
        self.validator.schema = AcceleratorValidator.SCHEMA
        self.data: dict[str, Any] = self.validator.normalized(data)
        self.is_valid = True
        self.accelerator_dirname = os.path.dirname(accelerator_path)

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined accelerator is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data)
        errors = self.validator.errors
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Validation outside of schema
        self.check_special_core_ids()
        self.validate_all_cores()

        # 2d mesh specific checks
        self.check_2d_mesh_schema()
        self.check_2d_mesh_layout()

        return self.is_valid

    def check_special_core_ids(self) -> None:
        """Check wether the special cores (pooling, simd, offchip) have a valid id"""

        pooling_core = self.data["graph"]["pooling_core_id"]
        if pooling_core is not None and pooling_core >= len(self.data["cores"]):
            self.invalidate(
                f"Specified pooling core id {self.data['graph']['pooling_core_id']} exceeds length of list of cores."
            )
        simd_core = self.data["graph"]["simd_core_id"]
        if simd_core is not None and simd_core >= len(self.data["cores"]):
            self.invalidate(
                f"Specified simd core id {self.data['graph']['simd_core_id']} exceeds length of list of cores."
            )
        offchip_core = self.data["graph"]["offchip_core_id"]
        if offchip_core is not None and offchip_core >= len(self.data["cores"]):
            self.invalidate(
                f"Specified offchip core id {self.data['graph']['offchip_core_id']} exceeds length of list of cores."
            )

    def check_2d_mesh_schema(self):
        if self.data["graph"]["type"] == "2d_mesh":
            if "nb_rows" not in self.data["graph"] or "nb_cols" not in self.data["graph"]:
                self.invalidate("graph of type 2d_mesh must contain 'nb_rows' and 'nb_cols'.")

    def check_2d_mesh_layout(self):
        if self.data["graph"]["type"] != "2d_mesh":
            return
        """Check wether the 2d mesh layout works with the given number of cores"""
        nb_special_cores = (
            (1 if self.data["graph"]["pooling_core_id"] is not None else 0)
            + (1 if self.data["graph"]["simd_core_id"] is not None else 0)
            + (1 if self.data["graph"]["offchip_core_id"] is not None else 0)
        )
        nb_regular_cores = len(self.data["cores"]) - nb_special_cores
        mesh_size = self.data["graph"]["nb_rows"] * self.data["graph"]["nb_cols"]
        if nb_regular_cores != mesh_size:
            self.invalidate(
                f"Number of cores (excl. pooling, simd, offchip) ({nb_regular_cores}) does not equal mesh size "
                f"({mesh_size})"
            )

    def validate_all_cores(self) -> None:
        """For all given core file paths:
        - parse core data
        - normalize core data (replace with defaults)
        - validate core data
        - replace core file path with core data
        """
        for core_id, core_file_name in enumerate(self.data["cores"]):
            core_data = self.open_core(core_file_name)
            # Stop validation if invalid core name is found
            if core_data is None:
                return

            core_validator = CoreValidator(core_data)
            validate_success = core_validator.validate()
            if not validate_success:
                self.invalidate(f"User-given core  {core_file_name} cannot be validated.")

            # Fill in default values
            normalized_core_data = core_validator.normalized_data
            self.data["cores"][core_id] = normalized_core_data

    def open_core(self, core_file_name: str) -> dict[str, Any] | None:
        """Find core with given yaml file name and read data."""
        if "./" in core_file_name:
            core_file_path = os.path.normpath(os.path.join(self.accelerator_dirname, core_file_name))
            return open_yaml(core_file_path)
        if "/" in core_file_name:
            return open_yaml(core_file_name)
        input_location = AcceleratorValidator.INPUT_DIR_LOCATION
        for dir_root_name, _, files_this_dir in os.walk(input_location):
            # Only consider subdirectories of `hardware` folder
            if "hardware" in dir_root_name:
                if core_file_name in files_this_dir:
                    core_file_path = dir_root_name + "/" + core_file_name
                    return open_yaml(core_file_path)

        self.invalidate(
            f"Core with filename `{core_file_name}` not found. Make sure `{input_location}` contains a folder "
            f"called `hardware` that contains the core file."
        )

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data
