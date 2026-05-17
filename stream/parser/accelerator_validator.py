import logging
import os
from functools import reduce
from itertools import combinations
from typing import Any

from cerberus import Validator
from zigzag.utils import open_yaml

from stream.parser.core_validator import ALLOWED_KINDS, ALLOWED_NAMESPACES, CoreValidatorRegistry

logger = logging.getLogger(__name__)


class AcceleratorValidator:
    INPUT_DIR_LOCATION = "stream/inputs/"
    FILENAME_REGEX = (
        r"^(?:\.\/)?"  # optional "./"
        r"(?:[A-Za-z0-9_\-]+\/)*"  # zero or more directories
        r"[A-Za-z0-9_\-]+"  # file name
        r"(?:\.ya?ml)?$"  # optional ".yaml" or ".yml"
    )
    CORE_IDS_REGEX = r"^\d+(?:\s*,\s*\d+){1,}$"
    COORDINATES_LEN = 2

    SCHEMA: dict[str, Any] = {
        # ------------------------------------------------------------------ #
        # Basic identification                                               #
        # ------------------------------------------------------------------ #
        "name": {"type": "string", "required": True},
        # ------------------------------------------------------------------ #
        # Core catalogue (file names relative to the inputs folder)          #
        # ------------------------------------------------------------------ #
        "cores": {
            "type": "dict",
            "required": True,
            "valuesrules": {"type": "string", "regex": FILENAME_REGEX},
        },
        # Id of the core that acts as the off-chip memory controller
        "offchip_core_id": {"type": "integer", "min": 0, "required": True},
        # ------------------------------------------------------------------ #
        # Optional unit_energy_cost that is used for all connections         #
        # that don't specify their own unit_energy_cost                      #
        # ------------------------------------------------------------------ #
        "unit_energy_cost": {
            "type": "float",
            "min": 0,
            "required": False,
            "default": 0,
        },
        # ------------------------------------------------------------------ #
        # Topology description                                               #
        # ------------------------------------------------------------------ #
        "core_connectivity": {
            "type": "list",
            "required": True,
            "schema": {
                "type": "dict",
                "schema": {
                    # "link" = point-to-point, "bus" = shared medium
                    "type": {
                        "type": "string",
                        "allowed": ["link", "bus"],
                        "default": "link",
                    },
                    # List of core ids that participate in this connection
                    "cores": {
                        "type": "list",
                        "minlength": 2,
                        "schema": {"type": "integer", "min": 0},
                    },
                    # Peak bandwidth for the link / bus (GB/s or chosen units)
                    "bandwidth": {"type": "float", "min": 0, "required": True},
                    # Optional override of the global unit energy cost
                    "unit_energy_cost": {"type": "float", "min": 0, "required": False},
                },
            },
        },
        # ------------------------------------------------------------------ #
        # Optional memory-sharing groups                                     #
        # ------------------------------------------------------------------ #
        "core_memory_sharing": {
            "type": "list",
            "default": [],
            "schema": {"type": "string", "regex": CORE_IDS_REGEX},
        },
        # ------------------------------------------------------------------ #
        # Translation from core id to coordinates                            #
        # ------------------------------------------------------------------ #
        "core_coordinates": {
            "type": "dict",
            "required": False,
            "default": {},
            "valuesrules": {"type": "list", "minlength": 2, "maxlength": 2, "schema": {"type": "integer"}},
        },
    }

    def __init__(self, data: Any, accelerator_path: str):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = Validator()
        self.validator.schema = AcceleratorValidator.SCHEMA  # type: ignore
        self.data: dict[str, Any] = self.validator.normalized(data)  # type: ignore
        self.is_valid = True
        self.accelerator_dirname = os.path.dirname(accelerator_path)
        self.errors: list[str] = []

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        self.errors.append(extra_msg)
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
        self.validate_namespace()
        self.validate_core_coordinates()

        self.validate_core_connectivity()
        self.validate_core_mem_sharing()

        if not self.is_valid and self.errors:
            logger.critical("Accelerator validation failed with %d issue(s).", len(self.errors))

        return self.is_valid

    def validate_core_ids(self):
        core_ids = list(self.data["cores"].keys())
        if not all(isinstance(core_id, int) and core_id >= 0 for core_id in core_ids):
            self.invalidate("Invalid core id in `cores`: id is not a positive integer.")
        if len(core_ids) != max(core_ids) + 1:
            self.invalidate("Invalid core id in `cores`: not all core ids in range are in use.")
        if self.data["offchip_core_id"] not in core_ids:
            self.invalidate("offchip_core_id does not correspond to any entry in `cores`.")

    def validate_all_cores(self) -> None:
        """For all given core file paths:
        - parse core data
        - normalize core data (replace with defaults)
        - validate core data
        - replace core file path with core data
        """
        for core_id, core_file_name in self.data["cores"].items():
            normalized_core_data = self.validate_single_core(core_file_name)
            if normalized_core_data:
                self.data["cores"][core_id] = normalized_core_data

    def validate_core_coordinates(self) -> None:
        """Validate the *format* of core coordinates when the field is present.

        Coordinates are optional at the base level; namespace-specific validators
        (e.g. :class:`AIE2AcceleratorNamespaceValidator`) enforce their presence
        when required by the namespace.
        """
        core_coordinates = self.data.get("core_coordinates", {})
        if not core_coordinates:
            return  # absent or empty – namespace validator handles presence checks
        for core_id, coordinates in core_coordinates.items():
            if not isinstance(core_id, int) or core_id < 0:
                self.invalidate(f"Invalid core id in core_coordinates: {core_id} is not a positive integer.")
            if core_id not in self.data["cores"]:
                self.invalidate(f"Core id {core_id} in core_coordinates does not exist in cores.")
            if len(coordinates) != self.COORDINATES_LEN or not all(isinstance(coord, int) for coord in coordinates):
                self.invalidate(f"Invalid coordinates for core id {core_id}: {coordinates}.")

    def validate_namespace(self) -> None:
        """Enforce a single consistent core namespace and run namespace-specific checks.

        Called after :meth:`validate_all_cores` so every core entry in
        ``self.data["cores"]`` is already a fully-normalized dict.
        """
        namespaces: set[str] = set()
        for core_id, core_data in self.data["cores"].items():
            if not isinstance(core_data, dict):
                continue  # core failed to load; error already recorded
            core_type = core_data.get("type", "")
            if "." not in core_type:
                self.invalidate(
                    f"Core {core_id} has type '{core_type}' without a namespace prefix. "
                    "All core types must follow the '<namespace>.<kind>' format "
                    f"(e.g. 'zigzag.compute', 'aie2.compute'). "
                    f"Allowed namespaces: {sorted(ALLOWED_NAMESPACES)}, "
                    f"allowed kinds: {sorted(ALLOWED_KINDS)}."
                )
            else:
                ns = core_type.split(".")[0]
                kind = core_type.split(".")[-1]
                if ns not in ALLOWED_NAMESPACES:
                    self.invalidate(
                        f"Core {core_id} has unknown namespace '{ns}'. "
                        f"Allowed namespaces: {sorted(ALLOWED_NAMESPACES)}."
                    )
                if kind not in ALLOWED_KINDS:
                    self.invalidate(
                        f"Core {core_id} has unknown kind '{kind}'. Allowed kinds: {sorted(ALLOWED_KINDS)}."
                    )
                namespaces.add(ns)

        if len(namespaces) > 1:
            self.invalidate(
                f"All cores in an accelerator must share the same namespace, "
                f"but found multiple namespaces: {sorted(namespaces)}. "
                "Mix-namespace accelerators are not supported."
            )
            return

        if not namespaces:
            return  # all cores failed to load; errors already recorded

        namespace = next(iter(namespaces))
        validator_cls = AcceleratorNamespaceValidatorRegistry.get(namespace)
        if validator_cls is None:
            supported = AcceleratorNamespaceValidatorRegistry.supported_namespaces()
            self.invalidate(
                f"Namespace '{namespace}' is not supported. "
                f"Supported namespaces: {', '.join(supported)}. "
                "To add support, create and register a new AcceleratorNamespaceValidator subclass."
            )
            return

        validator_cls(self.data, self.invalidate).validate()

    # Stream-level extension fields that are not known to namespace validators
    # (e.g. ZigZag) and must be stripped before validation then re-injected.
    _STREAM_EXTENSION_FIELDS: tuple[str, ...] = ("operator_types",)

    def validate_single_core(self, core_file_name: str) -> None | dict[str, Any]:
        core_data = self.open_core(core_file_name)
        # Stop validation if invalid core name is found
        if core_data is None:
            return

        # Extract Stream-level extension fields before namespace validation strips them.
        extension_fields = {k: core_data.pop(k) for k in self._STREAM_EXTENSION_FIELDS if k in core_data}

        raw_type = core_data.get("type")
        default_kind = raw_type if raw_type in ALLOWED_KINDS else "compute"
        normalized_type = CoreValidatorRegistry.normalize_core_type(
            raw_type,
            default_namespace=CoreValidatorRegistry.default_namespace,
            default_kind=default_kind,
        )
        validator_cls = CoreValidatorRegistry.get_validator(normalized_type)
        if validator_cls is None:
            supported_types = ", ".join(CoreValidatorRegistry.supported_types())
            self.invalidate(
                f"Core '{core_file_name}' has unsupported type '{normalized_type}'. Supported types: {supported_types}"
            )
            return

        core_validator = validator_cls(core_data)
        validate_success = core_validator.validate()
        if not validate_success:
            self.invalidate(f"User-given core {core_file_name} cannot be validated.")
            self.errors.extend(core_validator.errors)

        # Fill in default values and re-inject Stream-level extension fields.
        normalized_core_data = core_validator.normalized_data
        normalized_core_data.update(extension_fields)
        return normalized_core_data

    def open_core(self, core_file_name: str) -> dict[str, Any] | None:
        """Find core with given yaml file name and read data."""
        if "./" in core_file_name:
            core_file_path = os.path.normpath(os.path.join(self.accelerator_dirname, core_file_name))
            core_data = open_yaml(core_file_path)
            assert isinstance(core_data, dict), "Core data must be a dictionary."
            return core_data
        if "/" in core_file_name:
            core_data = open_yaml(core_file_name)
            assert isinstance(core_data, dict), "Core data must be a dictionary."
            return core_data
        input_location = AcceleratorValidator.INPUT_DIR_LOCATION
        for dir_root_name, _, files_this_dir in os.walk(input_location):
            # Only consider subdirectories of `hardware` folder
            if "hardware" in dir_root_name:
                if core_file_name in files_this_dir:
                    core_file_path = dir_root_name + "/" + core_file_name
                    core_data = open_yaml(core_file_path)
                    assert isinstance(core_data, dict), "Core data must be a dictionary."
                    return core_data

        self.invalidate(
            f"Core with filename `{core_file_name}` not found. Make sure `{input_location}` contains a folder "
            f"called `hardware` that contains the core file."
        )
        return None

    def validate_core_connectivity(self):
        connections = self.data["core_connectivity"]
        if connections == []:
            return  # empty graph is allowed

        core_ids = set(self.data["cores"].keys())
        for idx, conn in enumerate(connections):
            cores = conn["cores"]
            bw = conn["bandwidth"]
            ue = conn.get("unit_energy_cost", self.data.get("unit_energy_cost", 0))

            # basic semantic checks (most syntactic ones are done by Cerberus)
            if not all(cid in core_ids for cid in cores):
                self.invalidate(f"`core_connectivity[{idx}].cores` contains unknown core id.")
            if bw <= 0:
                self.invalidate(f"`core_connectivity[{idx}].bandwidth` must be > 0.")
            if ue < 0:
                self.invalidate(f"`core_connectivity[{idx}].unit_energy_cost` must be ≥ 0.")

            # normalise: store cores as an immutable tuple & fill in defaults
            conn["cores"] = tuple(cores)
            conn.setdefault("type", "link")
            conn.setdefault("unit_energy_cost", self.data.get("unit_energy_cost", 0))

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
        connectivity_groups = [set(conn["cores"]) for conn in self.data["core_connectivity"]]
        for mem_sharing_group in mem_sharing_groups:
            # Check each link within the mem_sharing_group
            for id_a, id_b in combinations(mem_sharing_group, 2):
                if any({id_a, id_b}.issubset(group) for group in connectivity_groups):
                    self.invalidate(
                        "Cores that share memory should must not be explicitly connected in `core_connectivity`"
                    )

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data


# =============================================================================
# Namespace-specific accelerator validation
# -----------------------------------------------------------------------------
# Every accelerator YAML uses cores that all belong to a single namespace
# (e.g. "zigzag" or "aie2").  The classes below encode what extra top-level
# fields and constraints are required for each namespace.
#
# HOW TO ADD A NEW NAMESPACE
# --------------------------
#   1. Create a subclass of BaseAcceleratorNamespaceValidator below.
#   2. Set NAMESPACE = "<your-namespace>" (must match the prefix used in
#      core_type strings, e.g. "aie3" for "aie3.compute").
#   3. Decorate the class with @AcceleratorNamespaceValidatorRegistry.register.
#   4. Override validate() and call self._invalidate(msg) for any violation.
#      The message will be logged and collected in the parent validator's error
#      list automatically.
# =============================================================================


class AcceleratorNamespaceValidatorRegistry:
    """Maps a namespace string to its accelerator-level namespace validator class."""

    _registry: dict[str, type["BaseAcceleratorNamespaceValidator"]] = {}

    @classmethod
    def register(cls, validator_cls: type["BaseAcceleratorNamespaceValidator"]):
        """Register *validator_cls* under its declared NAMESPACE."""
        cls._registry[validator_cls.NAMESPACE] = validator_cls
        return validator_cls

    @classmethod
    def get(cls, namespace: str) -> type["BaseAcceleratorNamespaceValidator"] | None:
        return cls._registry.get(namespace)

    @classmethod
    def supported_namespaces(cls) -> list[str]:
        return sorted(cls._registry.keys())


class BaseAcceleratorNamespaceValidator:
    """Base class for namespace-specific accelerator validators.

    Subclasses should set :attr:`NAMESPACE` and override :meth:`validate`.
    Violations are reported via :meth:`_invalidate`, which delegates to the
    parent :class:`AcceleratorValidator` so all errors are collected in one
    place.
    """

    NAMESPACE: str = ""  # must be overridden

    def __init__(self, data: dict[str, Any], invalidate_fn) -> None:
        self.data = data
        self._invalidate = invalidate_fn

    def validate(self) -> None:  # pragma: no cover
        """Override in subclasses to add namespace-specific validation."""


@AcceleratorNamespaceValidatorRegistry.register
class ZigZagAcceleratorNamespaceValidator(BaseAcceleratorNamespaceValidator):
    """Namespace validator for zigzag cores.  No extra top-level fields required."""

    NAMESPACE = "zigzag"

    def validate(self) -> None:
        pass  # zigzag accelerators have no namespace-specific requirements


@AcceleratorNamespaceValidatorRegistry.register
class AIE2AcceleratorNamespaceValidator(BaseAcceleratorNamespaceValidator):
    """Namespace validator for aie2 cores.

    Requires:
    - ``core_coordinates``: a non-empty mapping from core id to ``[col, row]``.
    """

    NAMESPACE = "aie2"

    def validate(self) -> None:
        self._validate_core_coordinates()

    def _validate_core_coordinates(self) -> None:
        coords = self.data.get("core_coordinates", {})
        if not coords:
            self._invalidate(
                "aie2 accelerators require a 'core_coordinates' section that maps every "
                "core id to its physical [col, row] position on the AIE array. "
                "Add 'core_coordinates' to your hardware YAML."
            )
