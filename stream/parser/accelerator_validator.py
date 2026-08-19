import copy
import logging
import os
from functools import reduce
from itertools import combinations
from typing import Any

from cerberus import Validator
from zigzag.utils import open_yaml

from stream.parser.core_validator import ALLOWED_KINDS, ALLOWED_NAMESPACES, CoreValidatorRegistry

logger = logging.getLogger(__name__)

INPUT_DIR_LOCATION = "stream/inputs/"

FILENAME_REGEX = (
    r"^(?:\.\/)?"  # optional "./"
    r"(?:[A-Za-z0-9_\-]+\/)*"  # zero or more directories
    r"[A-Za-z0-9_\-]+"  # file name
    r"(?:\.ya?ml)?$"  # optional ".yaml" or ".yml"
)


def resolve_core_path(core_file_name: str, accelerator_dirname: str) -> str | None:
    """Resolve a ``cores:`` reference to a path on disk, or None when nothing matches.

    Order: (1) explicit ``./`` or path-qualified refs taken as given; (2) a bare filename relative to
    the accelerator's own dir (``cores/`` then alongside); (3) a last-resort input-tree search that
    warns, being ambiguous when the name is reused under several ``hardware`` dirs.
    """
    if "./" in core_file_name:
        return os.path.normpath(os.path.join(accelerator_dirname, core_file_name))
    if "/" in core_file_name:
        return core_file_name

    # Bare filename: prefer the accelerator-local cores (deterministic).
    for candidate in (
        os.path.join(accelerator_dirname, "cores", core_file_name),
        os.path.join(accelerator_dirname, core_file_name),
    ):
        if os.path.isfile(candidate):
            return candidate

    # Fallback: search the input tree (legacy, ambiguous across hardware dirs).
    for dir_root_name, _, files_this_dir in os.walk(INPUT_DIR_LOCATION):
        if "hardware" in dir_root_name and core_file_name in files_this_dir:
            core_file_path = os.path.join(dir_root_name, core_file_name)
            logger.warning(
                "Core '%s' was not found next to its accelerator ('%s'); resolved via input-tree "
                "search to '%s'. Place core files in the accelerator's 'cores/' dir to avoid loading "
                "the wrong file when the name is reused elsewhere.",
                core_file_name,
                accelerator_dirname,
                core_file_path,
            )
            return core_file_path
    return None


class AcceleratorValidator:
    INPUT_DIR_LOCATION = INPUT_DIR_LOCATION
    FILENAME_REGEX = FILENAME_REGEX
    CORE_IDS_REGEX = r"^\d+(?:\s*,\s*\d+){1,}$"
    # "<core id>.<memory instance name>", e.g. "8.vmem"
    MEMORY_REF_REGEX = r"^\d+\.[A-Za-z0-9_\-]+$"
    COORDINATES_LEN = 2

    SCHEMA: dict[str, Any] = {
        # Basic identification
        "name": {"type": "string", "required": True},
        # Core catalogue (file names relative to the inputs folder)
        # A value is either a file reference (several ids may share one file) or a fully inlined
        # core description. The inline form is what a de-aliased `HardwareBundle` emits, so that a
        # mutation can target a single core id without the shared file dragging its siblings along.
        "cores": {
            "type": "dict",
            "required": True,
            "valuesrules": {
                "anyof": [
                    {"type": "string", "regex": FILENAME_REGEX},
                    {"type": "dict"},
                ]
            },
        },
        # Id of the core that acts as the off-chip memory controller
        "offchip_core_id": {"type": "integer", "min": 0, "required": True},
        # Optional unit_energy_cost used for connections that don't specify their own
        "unit_energy_cost": {
            "type": "float",
            "min": 0,
            "required": False,
            "default": 0,
        },
        # Topology description
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
        # Optional memory-sharing groups
        "core_memory_sharing": {
            "type": "list",
            "default": [],
            "schema": {"type": "string", "regex": CORE_IDS_REGEX},
        },
        # Translation from core id to coordinates
        "core_coordinates": {
            "type": "dict",
            "required": False,
            "default": {},
            "valuesrules": {"type": "list", "minlength": 2, "maxlength": 2, "schema": {"type": "integer"}},
        },
        # Physical-cost annotations (see stream.hardware.cost)
        # Process node the numbers below are meant to describe, e.g. "n3". Only the area/energy
        # model reads it; scheduling is unaffected.
        "technology_node": {"type": "string", "required": False},
        # Groups of "<core id>.<memory name>" references that are *views of one physical memory*.
        # Stream models a shared scratchpad as a separate memory per core that sees it, so without
        # this the cost model would bill the same silicon once per view. Costing-only: it changes
        # no scheduling behaviour.
        "memory_aliases": {
            "type": "list",
            "required": False,
            "default": [],
            "schema": {
                "type": "list",
                "minlength": 2,
                "schema": {"type": "string", "regex": MEMORY_REF_REGEX},
            },
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
        """Validate the accelerator data; log a critical warning when invalid and return True iff valid."""
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
        self.validate_memory_aliases()

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
        """For every core entry:
        - parse core data (a file reference is opened; an inline description is taken as given)
        - normalize core data (replace with defaults)
        - validate core data
        - replace the entry with the normalized core data
        """
        for core_id, core_entry in self.data["cores"].items():
            if isinstance(core_entry, dict):
                # Inline description: copy first, validation pops the Stream extension fields.
                normalized_core_data = self.validate_core_data(copy.deepcopy(core_entry), f"core {core_id}")
            else:
                normalized_core_data = self.validate_single_core(core_entry)
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
    _STREAM_EXTENSION_FIELDS: tuple[str, ...] = ("operator_types", "operand_precision")

    def validate_single_core(self, core_file_name: str) -> None | dict[str, Any]:
        core_data = self.open_core(core_file_name)
        # Stop validation if invalid core name is found
        if core_data is None:
            return
        return self.validate_core_data(core_data, core_file_name)

    def validate_core_data(self, core_data: dict[str, Any], label: str) -> None | dict[str, Any]:
        """Validate and normalize one core description. ``label`` only names it in error messages."""
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
                f"Core '{label}' has unsupported type '{normalized_type}'. Supported types: {supported_types}"
            )
            return

        core_validator = validator_cls(core_data)
        validate_success = core_validator.validate()
        if not validate_success:
            self.invalidate(f"User-given core {label} cannot be validated.")
            self.errors.extend(core_validator.errors)

        # Fill in default values and re-inject Stream-level extension fields.
        normalized_core_data = core_validator.normalized_data
        normalized_core_data.update(extension_fields)
        return normalized_core_data

    def open_core(self, core_file_name: str) -> dict[str, Any] | None:
        """Resolve a core reference to its YAML data. See :func:`resolve_core_path`."""
        core_file_path = resolve_core_path(core_file_name, self.accelerator_dirname)
        if core_file_path is None:
            self.invalidate(
                f"Core with filename `{core_file_name}` not found. Looked in "
                f"`{os.path.join(self.accelerator_dirname, 'cores')}` and under `{INPUT_DIR_LOCATION}`."
            )
            return None
        return self._read_core_yaml(core_file_path)

    @staticmethod
    def _read_core_yaml(core_file_path: str) -> dict[str, Any]:
        core_data = open_yaml(core_file_path)
        assert isinstance(core_data, dict), "Core data must be a dictionary."
        return core_data

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

    def validate_memory_aliases(self):
        """Every ``<core id>.<memory name>`` in `memory_aliases` must name a memory that exists.

        A typo here would silently *stop* deduplicating that memory and inflate the modelled area,
        which is exactly the kind of quiet cost error the budget guard exists to prevent.
        """
        for group in self.data.get("memory_aliases", []):
            for ref in group:
                core_id_str, _, mem_name = ref.partition(".")
                core_id = int(core_id_str)
                core_data = self.data["cores"].get(core_id)
                if core_data is None:
                    self.invalidate(f"`memory_aliases` entry '{ref}' names unknown core id {core_id}.")
                    continue
                if not isinstance(core_data, dict):
                    continue  # core failed to load; error already recorded
                known = set(core_data.get("memories", {}) or {})
                if "memory" in core_data:
                    known.add("memory")  # aie2 cores declare a single unnamed memory
                if mem_name not in known:
                    self.invalidate(
                        f"`memory_aliases` entry '{ref}' names memory '{mem_name}', which core "
                        f"{core_id} does not declare. Known: {sorted(known)}."
                    )

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data


# Namespace-specific accelerator validation: each namespace ("zigzag", "aie2") declares what extra
# top-level fields it requires. To add one, subclass BaseAcceleratorNamespaceValidator, set NAMESPACE,
# register it with @AcceleratorNamespaceValidatorRegistry.register, and override validate().


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
    """Base namespace validator: set :attr:`NAMESPACE`, override :meth:`validate`, report via ``_invalidate``."""

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
