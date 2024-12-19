import logging
import os
from functools import reduce
from itertools import combinations
from typing import Any

from cerberus import Validator
from zigzag.utils import open_yaml

from stream.parser.core_validator import CoreValidator

logger = logging.getLogger(__name__)


class AcceleratorValidator:
    INPUT_DIR_LOCATION = "stream/inputs/"
    GRAPH_TYPES = ["2d_mesh", "bus"]
    FILENAME_REGEX = r"^(?:[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-\///]+(\.yaml|\.yml))$"
    CORE_IDS_REGEX = r"^(\d+\s*,\s*)+\d+$"

    SCHEMA: dict[str, Any] = {
        "name": {"type": "string", "required": True},
        "cores": {
            "type": "dict",
            "required": True,
            "valuesrules": {"type": "string", "regex": FILENAME_REGEX},
        },
        "offchip_core": {"type": "string", "regex": FILENAME_REGEX, "required": False},
        "unit_energy_cost": {"type": "float", "default": 0},
        "bandwidth": {"type": "float", "required": True},
        "core_connectivity": {"type": "list", "required": True, "schema": {"type": "string", "regex": CORE_IDS_REGEX}},
        "core_memory_sharing": {
            "type": "list",
            "default": [],
            "schema": {"type": "string", "regex": CORE_IDS_REGEX},
        },
    }

    def __init__(self, data: Any, accelerator_path: str):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = Validator()
        self.validator.schema = AcceleratorValidator.SCHEMA  # type: ignore
        self.data: dict[str, Any] = self.validator.normalized(data)  # type: ignore
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
        validate_success = self.validator.validate(self.data)  # type: ignore
        errors = self.validator.errors  # type: ignore
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Validation outside of schema
        self.validate_core_ids()
        self.validate_all_cores()

        self.validate_core_connectivity()
        self.validate_core_mem_sharing()

        return self.is_valid

    def validate_core_ids(self):
        core_ids = list(self.data["cores"].keys())
        if not all(isinstance(core_id, int) and core_id >= 0 for core_id in core_ids):
            self.invalidate("Invalid core id in `cores`: id is not a positive integer.")
        if len(core_ids) != max(core_ids) + 1:
            self.invalidate("Invalid core id in `cores`: not all core ids in range are in use.")

    def validate_all_cores(self) -> None:
        """For all given core file paths:
        - parse core data
        - normalize core data (replace with defaults)
        - validate core data
        - replace core file path with core data
        """
        core_id = 0
        for core_id, core_file_name in self.data["cores"].items():
            normalized_core_data = self.validate_single_core(core_file_name)
            if not normalized_core_data:
                return
            self.data["cores"][core_id] = normalized_core_data

        # Offchip core (not part of cores list)
        if "offchip_core" in self.data:
            offchip_core_file_name = self.data["offchip_core"]
            normalized_core_data = self.validate_single_core(offchip_core_file_name)
            if not normalized_core_data:
                return
            self.data["offchip_core"] = normalized_core_data

    def validate_single_core(self, core_file_name: str) -> None | dict[str, Any]:
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
        return normalized_core_data

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

    def validate_core_connectivity(self):
        connectivity_data = self.data["core_connectivity"]

        # Empty data is okay
        if connectivity_data == []:
            return

        # Replace string of core ids with tuple of ints
        connectivity_groups = [tuple(int(i) for i in group.replace(" ", "").split(",")) for group in connectivity_data]
        self.data["core_connectivity"] = connectivity_groups

        all_connected_core_ids = reduce(lambda x, y: x + y, connectivity_groups)
        core_ids = list(self.data["cores"].keys())

        # Connection length >= 2
        if not all(len(group) > 1 for group in connectivity_groups):
            self.invalidate("Core connection should contain at least 2 core ids.")

        # No unknown core ids
        if not all(connected_id in core_ids for connected_id in all_connected_core_ids):
            self.invalidate("`core_connectivity` contains unknown core id.")

    def validate_core_mem_sharing(self):
        # Replace string of core ids with tuple of ints
        mem_sharing_data = self.data["core_memory_sharing"]
        if len(mem_sharing_data) == 0:
            return
        mem_sharing_groups = [tuple(int(i) for i in group.replace(" ", "").split(",")) for group in mem_sharing_data]
        self.data["core_memory_sharing"] = mem_sharing_groups

        all_mem_sharing_ids = reduce(lambda x, y: x + y, mem_sharing_groups)
        core_ids = list(self.data["cores"].keys())

        # Connection length >= 2
        if not all(len(group) > 1 for group in mem_sharing_groups):
            self.invalidate("Shared memory connection should contain at least 2 core ids.")

        # No unknown core ids
        if not all(mem_sharing_id in core_ids for mem_sharing_id in all_mem_sharing_ids):
            self.invalidate("`core_memory_sharing` contains unknown core id.")

        # Cores that share memory should not have an explicit connection
        connectivity_groups = self.data["core_connectivity"]
        for mem_sharing_group in mem_sharing_groups:
            # Check each link within the mem_sharing_group
            for id_a, id_b in combinations(mem_sharing_group, 2):
                if any(
                    id_a in connectivity_group and id_b in connectivity_group
                    for connectivity_group in connectivity_groups
                ):
                    self.invalidate(
                        "Cores that share memory should must not be explicitly connected in `core_connectivity`"
                    )

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data
