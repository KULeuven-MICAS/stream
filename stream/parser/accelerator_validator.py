import copy
import logging
import os
import re
from typing import Any

from cerberus import Validator
from zigzag.utils import open_yaml

from stream.datatypes import ELEMENT_BITS
from stream.hardware.ports import MemoryView, PortKey
from stream.parser.core_validator import ALLOWED_KINDS, ALLOWED_NAMESPACES, CoreValidatorRegistry

logger = logging.getLogger(__name__)

INPUT_DIR_LOCATION = "stream/inputs/"

FILENAME_REGEX = (
    r"^(?:\.\/)?"  # optional "./"
    r"(?:[A-Za-z0-9_\-]+\/)*"  # zero or more directories
    r"[A-Za-z0-9_\-]+"  # file name
    r"(?:\.ya?ml)?$"  # optional ".yaml" or ".yml"
)

# The name of a physical memory (`memories`), e.g. "l1_c0"
MEMORY_NAME_REGEX = r"^[A-Za-z_][A-Za-z0-9_\-]*$"
# "<memory>.<port name>", e.g. "dram.rw_port_1": one port of a memory
PORT_REF_REGEX = r"^[A-Za-z_][A-Za-z0-9_\-]*\.[A-Za-z0-9_\-]+$"
# "<core id>.<memory instance name>", e.g. "8.vmem": a core's view of a memory, its top-level memory instance
MEMORY_VIEW_REGEX = r"^\d+\.[A-Za-z0-9_\-]+$"
# A port's name within its memory, e.g. "rw_port_1"
PORT_NAME_REGEX = r"^[A-Za-z0-9_\-]+$"

# A port that is a port the backends of `cores` declare on their views of the memory, by that port's name
VIEW_PORT_SCHEMA: dict[str, Any] = {
    "from": {"type": "string", "required": True, "regex": PORT_NAME_REGEX},
    "cores": {"type": "list", "required": True, "minlength": 1, "schema": {"type": "integer", "min": 0}},
}
# A port of the memory's own, as an off-chip memory's: its width in bits per cycle, the directions it serves per operand
# role ("input<k>", "output", or "any" for every operand), and its energies in pJ per bit
OWN_PORT_SCHEMA: dict[str, Any] = {
    "bandwidth": {"type": "number", "required": True, "min": 0},
    "serves": {
        "type": "dict",
        "required": True,
        "minlength": 1,
        "keysrules": {"type": "string", "regex": r"^(any|output|input\d+)$"},
        "valuesrules": {"type": "list", "minlength": 1, "schema": {"type": "string", "allowed": ["read", "write"]}},
    },
    "read_energy": {"type": "number", "required": True, "min": 0},
    "write_energy": {"type": "number", "required": True, "min": 0},
}


def parse_port_ref(ref: str) -> PortKey:
    """Split a ``bandwidth:`` port key into (memory, port name)."""
    memory, port = ref.split(".")
    return PortKey(memory, port)


def parse_memory_view(view: str) -> MemoryView:
    """Split a ``memories:`` entry ``<core id>.<memory instance name>``."""
    core, _, instance = view.partition(".")
    return int(core), instance


def resolve_core_path(core_file_name: str, accelerator_dirname: str) -> str | None:
    """Resolve a ``cores:`` reference to a path on disk, or None when nothing matches."""
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
    COORDINATES_LEN = 2

    SCHEMA: dict[str, Any] = {
        # Basic identification
        "name": {"type": "string", "required": True},
        "kernel_library": {"type": "string", "required": False},
        # Core catalogue: each value is either a file reference (ids may share one file) or an
        # inlined core description.
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
        # Memory name -> measured bandwidth every transfer into or out of that memory shares, or a port key -> measured
        # bandwidth replacing that top-level port's declared width
        "bandwidth": {
            "type": "dict",
            "required": False,
            "default": {},
            "keysrules": {
                "anyof": [
                    {"type": "string", "regex": PORT_REF_REGEX},
                    {"type": "string", "regex": MEMORY_NAME_REGEX},
                ]
            },
        },
        # The physical memories, by name. No core owns one: its views are the top-level memory instance of each core
        # that uses it, as "<core id>.<instance>", and a core views one memory per top-level instance. A memory no core
        # views declares its capacity in bits. Its ports, by name, are each one it declares or one its views' cores
        # declare.
        "memories": {
            "type": "dict",
            "required": True,
            "keysrules": {"type": "string", "regex": MEMORY_NAME_REGEX},
            "valuesrules": {
                "type": "dict",
                "schema": {
                    "offchip": {"type": "boolean", "required": True},
                    "views": {
                        "type": "list",
                        "required": True,
                        "schema": {"type": "string", "regex": MEMORY_VIEW_REGEX},
                    },
                    "capacity_bits": {"type": "integer", "min": 1},
                    "ports": {
                        "type": "dict",
                        "required": True,
                        "keysrules": {"type": "string", "regex": PORT_NAME_REGEX},
                        "valuesrules": {"type": "dict", "anyof_schema": [VIEW_PORT_SCHEMA, OWN_PORT_SCHEMA]},
                    },
                },
            },
        },
        # What reconfiguring the array between designs costs, read by the namespace constraints
        "reconfiguration": {"type": "dict", "required": False},
        # Optional unit_energy_cost used for connections that don't specify their own
        "unit_energy_cost": {
            "type": "float",
            "min": 0,
            "required": False,
            "default": 0,
        },
        # The links and buses that move data between memories
        "connections": {
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
                    # The memories this connection joins, by name, each with the ports of it the connection enters and
                    # leaves it through
                    "ends": {
                        "type": "dict",
                        "minlength": 2,
                        "required": True,
                        "keysrules": {"type": "string", "regex": MEMORY_NAME_REGEX},
                        "valuesrules": {"type": "list", "schema": {"type": "string", "regex": PORT_NAME_REGEX}},
                    },
                    # Peak bandwidth for the link / bus (GB/s or chosen units)
                    "bandwidth": {"type": "float", "min": 0, "required": True},
                    # Optional override of the global unit energy cost
                    "unit_energy_cost": {"type": "float", "min": 0, "required": False},
                },
            },
        },
        # Translation from core id to coordinates
        "core_coordinates": {
            "type": "dict",
            "required": False,
            "default": {},
            "valuesrules": {"type": "list", "minlength": 2, "maxlength": 2, "schema": {"type": "integer"}},
        },
        # Physical-cost annotations (see stream.hardware.cost); scheduling-inert.
        # Process node the numbers below describe, e.g. "n3".
        "technology_node": {"type": "string", "required": False},
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
            return False  # the checks below read what the schema guarantees

        # Validation outside of schema
        self.validate_core_ids()
        self.validate_kernel_library()
        self.validate_all_cores()
        self.validate_namespace()
        self.validate_core_coordinates()

        self.validate_memories()
        self.validate_connections()
        self.validate_bandwidth()

        if not self.is_valid and self.errors:
            logger.critical("Accelerator validation failed with %d issue(s).", len(self.errors))

        return self.is_valid

    def validate_kernel_library(self) -> None:
        if (path := self.data.get("kernel_library")) is None:
            return
        path = os.path.normpath(os.path.join(self.accelerator_dirname, path))
        if not os.path.isfile(path):
            self.invalidate(f"kernel_library {path} does not exist.")
        self.data["kernel_library"] = path

    def validate_core_ids(self):
        core_ids = list(self.data["cores"].keys())
        if not all(isinstance(core_id, int) and core_id >= 0 for core_id in core_ids):
            self.invalidate("Invalid core id in `cores`: id is not a positive integer.")

    def validate_all_cores(self) -> None:
        """Parse, normalize and validate each core entry (file reference or inline dict) into normalized data."""
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
        if "banks" in core_data:
            core_data.pop("banks")
            self.invalidate(f"Core {label}: `banks` belongs to the memory level it banks, as `memories.<level>.banks`.")
        # Each memory level's banks, a Stream field the namespace validators do not know
        levels = {"memory": core_data["memory"]} if "memory" in core_data else core_data.get("memories") or {}
        memory_banks = {
            name: level.pop("banks") for name, level in levels.items() if isinstance(level, dict) and "banks" in level
        }

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

        precision = extension_fields.get("operand_precision") or {}
        if not isinstance(precision, dict) or not set(precision) <= {"input", "accumulator"}:
            self.invalidate(f"Core {label}: `operand_precision` takes only `input` and `accumulator`.")
        elif unknown := {str(v) for v in precision.values()} - ELEMENT_BITS.keys():
            self.invalidate(f"Core {label}: unknown element types {sorted(unknown)} in `operand_precision`.")

        for name, banks in memory_banks.items():
            if (
                not isinstance(banks, dict)
                or not {"count", "width"} <= set(banks) <= {"count", "width", "stripe"}
                or any(type(v) is not int or v <= 0 for v in banks.values())
            ):
                self.invalidate(
                    f"Core {label}: memory `{name}`'s `banks` takes whole positive `count`, `width` in bytes and "
                    "optional `stripe` in words."
                )

        # Fill in default values and re-inject Stream-level extension fields.
        normalized_core_data = core_validator.normalized_data
        normalized_core_data.update(extension_fields)
        normalized_core_data["memory_banks"] = memory_banks
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

    def validate_connections(self):
        """Each connection joins known memories, a link exactly two, through ports each end's memory declares, and
        every memory has one: a memory no connection reaches holds nothing a transfer can bring. A memory appears once
        per connection, as its ends are keyed by it. A connection without its own unit energy cost takes the
        accelerator's."""
        connections = self.data["connections"]
        for idx, conn in enumerate(connections):
            ends = conn["ends"]
            if unknown := [m for m in ends if m not in self.data["memories"]]:
                self.invalidate(f"`connections[{idx}].ends` names unknown memories {unknown}.")
            if conn["type"] == "link" and len(ends) != 2:  # noqa: PLR2004
                self.invalidate(f"`connections[{idx}]` is a link, which joins exactly two memories, not {list(ends)}.")
            for memory, ports in ends.items():
                declared = self.data["memories"].get(memory, {}).get("ports", {})
                if memory in self.data["memories"] and (undeclared := sorted(set(ports) - set(declared))):
                    self.invalidate(f"`connections[{idx}].ends.{memory}` names ports {undeclared} it does not declare.")
                if len(set(ports)) != len(ports):
                    self.invalidate(f"`connections[{idx}].ends.{memory}` names a port twice: {ports}.")
            if conn["bandwidth"] <= 0:
                self.invalidate(f"`connections[{idx}].bandwidth` must be > 0.")
            conn["ends"] = {memory: tuple(ports) for memory, ports in ends.items()}
            conn.setdefault("unit_energy_cost", self.data["unit_energy_cost"])
        connected = {name for conn in connections for name in conn["ends"]}
        if unreached := [name for name in self.data["memories"] if name not in connected]:
            self.invalidate(f"memories {unreached} have no connection: nothing can reach them.")

    def validate_bandwidth(self) -> None:
        """Every ``bandwidth:`` key names an existing memory, or a port of one without a measured model; the
        accelerator factory checks that a port key names a modelled port."""
        keys = self.data["bandwidth"]
        for key, model in keys.items():
            self._validate_bandwidth_model(key, model)
        for key in keys:
            memory = parse_port_ref(key).memory if re.match(PORT_REF_REGEX, key) else key
            if memory not in self.data["memories"]:
                self.invalidate(f"`bandwidth.{key}` names unknown memory {memory}.")
            elif memory != key and memory in keys:
                self.invalidate(f"`bandwidth.{key}`: memory {memory} already has a measured bandwidth.")

    def _validate_bandwidth_model(self, key: object, model: object) -> None:
        """A bandwidth model: positive ``ceiling`` and ``contiguous`` rates, measured ``strided`` rates per direction,
        and a positive whole number of bytes per access as ``burst``, which a direction's measured rates override."""
        if not isinstance(model, dict):
            self.invalidate(f"`bandwidth.{key}` must be a mapping.")
            return
        for rate in ("ceiling", "contiguous"):
            if not isinstance(model.get(rate), int | float) or model[rate] <= 0:
                self.invalidate(f"`bandwidth.{key}.{rate}` must be a positive number.")
        for direction, table in (model.get("strided") or {}).items():
            if direction not in ("read", "write") or not isinstance(table, dict):
                self.invalidate(f"`bandwidth.{key}.strided` maps `read` and `write` to rates per span in bytes.")
        if "burst" in model and (type(model["burst"]) is not int or model["burst"] <= 0):
            self.invalidate(f"`bandwidth.{key}.burst` must be a positive whole number of bytes.")

    def validate_memories(self):
        """Every view names a core and an instance that core declares, no instance is a view of two memories, the views
        of a memory agree on its size, a memory no core views declares its capacity and one viewed declares none, a
        port's cores view its memory, at most one memory is off-chip, and every core sees a memory; views normalised to
        (core id, instance) pairs. That the views are the core's top levels, all of them, and that a memory's ports
        cover the ports its views' cores declare, is the factory's to check: the backend knows both."""
        views: dict[int, list[str]] = {core_id: [] for core_id in self.data["cores"]}
        bound: dict[tuple[int, str], str] = {}
        for name, memory in self.data["memories"].items():
            parsed = [parse_memory_view(entry) for entry in memory["views"]]
            sizes: dict[str, int] = {}
            for core_id, instance in parsed:
                if core_id not in self.data["cores"]:
                    self.invalidate(f"`memories.{name}` has a view of unknown core {core_id}.")
                    continue
                views[core_id].append(name)
                if (other := bound.get((core_id, instance))) is not None:
                    self.invalidate(f"`{core_id}.{instance}` is a view of both {other} and {name}.")
                bound[(core_id, instance)] = name
                core_data = self.data["cores"][core_id]
                if not isinstance(core_data, dict):
                    continue  # the core failed to load; that error is recorded already
                declared = {"memory": core_data["memory"]} if "memory" in core_data else core_data["memories"]
                if instance not in declared:
                    self.invalidate(
                        f"`memories.{name}`: core {core_id} declares no memory `{instance}`. Known: {sorted(declared)}."
                    )
                    continue
                size = declared[instance].get("capacity" if "memory" in core_data else "size")
                if size is not None:  # a core missing it failed its own validation already
                    sizes[f"{core_id}.{instance}"] = size
            if len(set(sizes.values())) > 1:
                self.invalidate(f"`memories.{name}`: its views are one memory but declare different sizes {sizes}.")
            self._validate_capacity_and_ports(name, memory, {core_id for core_id, _ in parsed})
            memory["views"] = tuple(parsed)
        if len(offchip := [name for name, memory in self.data["memories"].items() if memory["offchip"]]) > 1:
            self.invalidate(f"memories {offchip} are all off-chip; an accelerator has at most one off-chip memory.")
        for core_id, names in views.items():
            if not names:
                self.invalidate(f"core {core_id} is a view of no memory.")

    def _validate_capacity_and_ports(self, name: str, memory: dict[str, Any], viewers: set[int]) -> None:
        """A memory no core views declares its capacity and one viewed declares none, and a port's cores view it."""
        if viewers and "capacity_bits" in memory:
            self.invalidate(f"`memories.{name}`: its views declare its capacity, so `capacity_bits` is not given.")
        if not viewers and "capacity_bits" not in memory:
            self.invalidate(f"`memories.{name}`: no core views it, so it declares `capacity_bits`.")
        for port_name, port in memory["ports"].items():
            if strangers := sorted(set(port.get("cores", ())) - viewers):
                self.invalidate(f"`memories.{name}.ports.{port_name}` names cores {strangers}, no views of it.")

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
