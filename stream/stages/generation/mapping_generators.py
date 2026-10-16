"""Mapping generators: each proposes a mapping for the accelerators it claims when the caller gives none.

A generator registers under the ``stream.mapping_generators`` entry-point group with a ``name``, a
``priority``, a ``claims(accelerator)`` predicate and ``stages()``, the stages that turn the workload
in the context into ``sub_workloads`` and ``sub_mappings``, one per fused group.
"""

from __future__ import annotations

from typing import Protocol

from stream.hardware.architecture.accelerator import Accelerator
from stream.plugins import claimant
from stream.stages.stage import StageCallable

MAPPING_GENERATORS_GROUP = "stream.mapping_generators"


class MappingGenerator(Protocol):
    name: str
    priority: int

    def claims(self, accelerator: Accelerator) -> bool: ...

    def stages(self) -> list[StageCallable]: ...


class GenericMappingGenerator:
    """Any accelerator: normalizations expanded into affine sub-ops, fused at the workload's own cut points."""

    name = "generic"
    priority = 0

    def claims(self, accelerator: Accelerator) -> bool:  # noqa: ARG002 -- the fallback for every accelerator
        return True

    def stages(self) -> list[StageCallable]:
        from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage  # noqa: PLC0415
        from stream.stages.generation.normalization_expansion import ExpandNormalizationStage  # noqa: PLC0415

        return [ExpandNormalizationStage, GenericMappingGenerationStage]


GENERIC_MAPPING = GenericMappingGenerator()


def mapping_generator_for(accelerator: Accelerator) -> MappingGenerator:
    return claimant(MAPPING_GENERATORS_GROUP, accelerator)
