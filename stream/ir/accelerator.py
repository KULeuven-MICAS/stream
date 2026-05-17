"""AcceleratorIR Pydantic model with per-persona view methods.

Wraps the output of Accelerator.get_ir() in a typed, versioned Pydantic model.
Construction is always via the from_internal() classmethod.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

_CORE_COMMON_FIELDS = {"id", "name", "core_type", "type", "row_id", "col_id", "utilization"}


class CoreIR(BaseModel):
    """IR representation of a single accelerator core.

    Common fields cover all core types. Type-specific fields (e.g. memory capacity for aie2,
    operational_array for zigzag) are captured in extra_fields as a dict. This avoids requiring
    a discriminated union over all possible core schemas at Phase 16 scope.
    """

    id: int = Field(description="Core unique identifier (integer index)")
    name: str = Field(description="Core human-readable name")
    core_type: str = Field(description="Core type namespace, e.g. 'aie2.compute' or 'zigzag.compute'")
    type: str = Field(description="Core role: 'compute', 'offchip', or 'dma'")
    row_id: int = Field(description="Row position in the 2-D core grid (-1 for offchip)")
    col_id: int = Field(description="Column position in the 2-D core grid (-1 for offchip)")
    utilization: float = Field(description="Core utilization ratio in [0, 1]")
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific fields: memory/fifo depth for aie2, operational_array for zigzag, etc.",
    )


class AcceleratorHardwareView(BaseModel):
    """Hardware-persona projection of AcceleratorIR.

    Contains per-core resource information (memory capacity, fifo depth, utilization)
    and full connectivity. Suitable for hardware engineers reasoning about resource budgets.
    """

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="Accelerator name")
    num_cores: int = Field(description="Total number of cores including offchip")
    cores: list[CoreIR] = Field(description="All cores with full resource info (including extra_fields)")
    core_connectivity: list[dict[str, Any]] = Field(description="Bus and link connectivity entries between cores")


class AcceleratorCompilerView(BaseModel):
    """Compiler-persona projection of AcceleratorIR.

    Contains core topology info (id, core_type, row_id, col_id) and connectivity.
    Suitable for compiler engineers performing placement and routing decisions.
    """

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="Accelerator name")
    cores: list[CoreIR] = Field(description="All cores — use id, core_type, row_id, col_id for placement")
    core_connectivity: list[dict[str, Any]] = Field(description="Bus and link connectivity entries between cores")


class AcceleratorIR(BaseModel):
    """Typed Pydantic model wrapping Accelerator.get_ir() output.

    schema_version '1.0': minor bumps (1.1) for additive fields, major bumps (2.0) for
    removed/renamed fields. Construction is always via from_internal().
    """

    model_config = ConfigDict(
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "stream_aie/accelerator_ir/v1",
        }
    )

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="Accelerator name as declared in hardware YAML")
    num_cores: int = Field(description="Total number of cores (including offchip core if present)")
    offchip_core_id: int | None = Field(
        description="ID of the offchip memory core, or None if the accelerator has no offchip core"
    )
    nb_shared_mem_groups: int = Field(description="Number of shared memory groups in the accelerator")
    cores: list[CoreIR] = Field(description="All cores with common and type-specific fields")
    core_connectivity: list[dict[str, Any]] = Field(
        description="Connectivity entries: bus (bidirectional, multiple cores) or link (directed pair)"
    )

    @classmethod
    def from_internal(cls, accelerator: Accelerator) -> AcceleratorIR:
        """Construct AcceleratorIR from an Accelerator internal object.

        Calls accelerator.get_ir() once. For each core dict, separates common fields
        (id, name, core_type, type, row_id, col_id, utilization) from type-specific fields
        which are collected into extra_fields.
        """

        raw = accelerator.get_ir()
        cores = []
        for c in raw["cores"]:
            extra = {k: v for k, v in c.items() if k not in _CORE_COMMON_FIELDS}
            cores.append(
                CoreIR(
                    id=c["id"],
                    name=c["name"],
                    core_type=c["core_type"],
                    type=c["type"],
                    row_id=c["row_id"],
                    col_id=c["col_id"],
                    utilization=c["utilization"],
                    extra_fields=extra,
                )
            )
        return cls(
            name=raw["name"],
            num_cores=raw["num_cores"],
            offchip_core_id=raw["offchip_core_id"],
            nb_shared_mem_groups=raw["nb_shared_mem_groups"],
            cores=cores,
            core_connectivity=raw["core_connectivity"],
        )

    def hardware_view(self) -> AcceleratorHardwareView:
        """Return hardware-persona projection: full core resource info and connectivity."""
        return AcceleratorHardwareView(
            name=self.name,
            num_cores=self.num_cores,
            cores=self.cores,
            core_connectivity=self.core_connectivity,
        )

    def compiler_view(self) -> AcceleratorCompilerView:
        """Return compiler-persona projection: core topology (id/type/position) and connectivity."""
        return AcceleratorCompilerView(
            name=self.name,
            cores=self.cores,
            core_connectivity=self.core_connectivity,
        )
