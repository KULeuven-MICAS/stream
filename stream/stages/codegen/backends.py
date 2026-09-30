"""Code generation backends: each emits the design of every fused group for the accelerators it claims.

A backend registers under the ``stream.codegen_backends`` entry-point group with a ``name``, a
``priority``, a ``claims(accelerator)`` predicate and a ``stage()`` returning the stage that generates
one fused group's code. That stage wraps the group's allocation stages and lowers what they solve.
"""

from __future__ import annotations

from typing import Protocol

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.plugins import claimant
from stream.stages.stage import StageCallable

CODEGEN_BACKENDS_GROUP = "stream.codegen_backends"


class CodegenBackend(Protocol):
    name: str
    priority: int

    def claims(self, accelerator: Accelerator) -> bool: ...

    def stage(self) -> StageCallable: ...


class AIE2CodegenBackend:
    """MLIR for AIE2 arrays, through the stream and aie dialects."""

    name = "aie2"
    priority = 10

    def claims(self, accelerator: Accelerator) -> bool:
        return any(isinstance(core, Core) and core.namespace == "aie2" for core in accelerator.core_list)

    def stage(self) -> StageCallable:
        from stream.stages.codegen.aie_code_generation import AIECodeGenerationStage  # noqa: PLC0415

        return AIECodeGenerationStage


AIE2_CODEGEN = AIE2CodegenBackend()


def codegen_backend_for(accelerator: Accelerator) -> CodegenBackend:
    return claimant(CODEGEN_BACKENDS_GROUP, accelerator)
