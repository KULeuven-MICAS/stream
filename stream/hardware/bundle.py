"""Per-core hardware bundles: the de-aliased accelerator plus one core description per core id."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator, resolve_core_path

__all__ = ["CORES_SUBDIR", "DEFAULT_ACCELERATOR_FILENAME", "HardwareBundle"]

DEFAULT_ACCELERATOR_FILENAME = "accelerator.yaml"
CORES_SUBDIR = "cores"


@dataclass
class HardwareBundle:
    """An accelerator plus one independent core description per core id."""

    name: str
    accelerator: dict[str, Any]
    cores: dict[int, dict[str, Any]]
    core_sources: dict[int, str] = field(default_factory=dict)
    accelerator_dirname: str = ""

    # Construction

    @classmethod
    def from_yaml(cls, accelerator_path: str | Path) -> HardwareBundle:
        """Read an accelerator YAML and de-alias its cores."""
        accelerator_path = str(accelerator_path)
        data = open_yaml(accelerator_path)
        if not isinstance(data, dict):
            raise ValueError(f"Accelerator file {accelerator_path} does not contain a mapping.")
        return cls.from_data(data, os.path.dirname(accelerator_path))

    @classmethod
    def from_data(cls, data: dict[str, Any], accelerator_dirname: str = "") -> HardwareBundle:
        """De-alias an already-loaded accelerator description."""
        data = copy.deepcopy(data)
        raw_cores: dict[int, Any] = data.pop("cores")
        cores: dict[int, dict[str, Any]] = {}
        sources: dict[int, str] = {}
        for core_id, entry in raw_cores.items():
            if isinstance(entry, dict):
                cores[core_id] = copy.deepcopy(entry)
                sources[core_id] = str(entry.get("name", f"core_{core_id}"))
                continue
            core_path = resolve_core_path(entry, accelerator_dirname)
            if core_path is None:
                raise FileNotFoundError(f"Core '{entry}' referenced by core id {core_id} could not be resolved.")
            core_data = open_yaml(core_path)
            if not isinstance(core_data, dict):
                raise ValueError(f"Core file {core_path} does not contain a mapping.")
            # The deep copy is the whole point: two ids reading one file must not share a dict.
            cores[core_id] = copy.deepcopy(core_data)
            sources[core_id] = Path(core_path).stem
        return cls(
            name=data["name"],
            accelerator=data,
            cores=cores,
            core_sources=sources,
            accelerator_dirname=accelerator_dirname,
        )

    def copy(self) -> HardwareBundle:
        """A fully independent copy — the starting point for any mutation."""
        return copy.deepcopy(self)

    # Cost-model annotations

    @property
    def technology_node(self) -> str | None:
        """Process node declared for this bundle, or None when the author did not say."""
        node = self.accelerator.get("technology_node")
        return str(node).lower() if node else None

    @property
    def memory_aliases(self) -> list[list[str]]:
        """Groups of ``<core id>.<memory name>`` refs that are views of one physical memory."""
        return [list(group) for group in self.accelerator.get("memory_aliases", []) or []]

    # Output forms

    def to_data(self) -> dict[str, Any]:
        """The accelerator description with every core inlined — what the validator consumes."""
        data = copy.deepcopy(self.accelerator)
        data["cores"] = {core_id: copy.deepcopy(core) for core_id, core in sorted(self.cores.items())}
        return data

    def validated_data(self) -> dict[str, Any]:
        """:meth:`to_data` run through the validators, so schema defaults are filled in."""
        validator = AcceleratorValidator(self.to_data(), os.path.join(self.accelerator_dirname, "accelerator.yaml"))
        normalized = validator.normalized_data
        if not validator.validate():
            raise ValueError(
                f"Hardware bundle '{self.name}' is not a valid accelerator:\n" + "\n".join(validator.errors)
            )
        return normalized

    def to_accelerator(self) -> Accelerator:
        """Build the in-memory :class:`Accelerator`, no file IO."""
        return AcceleratorFactory(self.validated_data()).create()

    def materialize(self, out_dir: str | Path, accelerator_filename: str = DEFAULT_ACCELERATOR_FILENAME) -> Path:
        """Write the bundle as an accelerator YAML plus one de-aliased core YAML per core id."""
        out_dir = Path(out_dir)
        cores_dir = out_dir / CORES_SUBDIR
        cores_dir.mkdir(parents=True, exist_ok=True)

        data = copy.deepcopy(self.accelerator)
        core_refs: dict[int, str] = {}
        for core_id, core in sorted(self.cores.items()):
            filename = self.core_filename(core_id)
            (cores_dir / filename).write_text(yaml.safe_dump(core, sort_keys=False))
            core_refs[core_id] = f"./{CORES_SUBDIR}/{filename}"
        data["cores"] = core_refs

        accelerator_path = out_dir / accelerator_filename
        accelerator_path.write_text(yaml.safe_dump(data, sort_keys=False))
        return accelerator_path

    def core_filename(self, core_id: int) -> str:
        """File name a materialized core gets. The id prefix is what keeps de-aliased siblings apart."""
        stem = self.core_sources.get(core_id) or str(self.cores[core_id].get("name", f"core_{core_id}"))
        # `AcceleratorValidator.FILENAME_REGEX` only accepts word characters and dashes.
        safe_stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
        return f"core_{core_id:02d}_{safe_stem}.yaml"
