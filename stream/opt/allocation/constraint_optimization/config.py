from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from zigzag.datatypes import MemoryOperand


class CoreRole(Enum):
    COMPUTE = auto()
    CACHE = auto()
    IO = auto()
    SHIM = auto()


@dataclass(frozen=True)
class DmaBudget:
    mem_to_dram: int = 0
    dram_to_mem: int = 0
    shim_s2mm: int = 0
    shim_mm2s: int = 0


@dataclass(frozen=True)
class BufferingConfig:
    double_buffer_factor: int = 1

    def __post_init__(self) -> None:
        if self.double_buffer_factor < 1:
            raise ValueError("double_buffer_factor must be >= 1")


@dataclass(frozen=True)
class IoRoutingRules:
    force_via_cache: bool = False
    max_cols_to_use: int | None = None  # e.g., cap AIE column usage


@dataclass(frozen=True)
class ObjectiveWeights:
    idle_weight: float = 0.0


@dataclass(frozen=True)
class CoreConstraintProfile:
    """
    Per-core-type constraint knobs. Mapped by core.type.
    """

    type_name: str
    roles: set[CoreRole]
    capacity_operand: MemoryOperand | None = None
    dma_limits: DmaBudget = field(default_factory=DmaBudget)
    buffering: BufferingConfig = field(default_factory=BufferingConfig)
    io_routing: IoRoutingRules = field(default_factory=IoRoutingRules)

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError(f"CoreConstraintProfile {self.type_name} must declare at least one role")


@dataclass(frozen=True)
class TransferMilpConfig:
    # Currently unused in Phase 1; defaults mirror existing behavior.
    nb_cols_to_use: int = 4
    force_io_via_cache: bool = True
    mem_dma_channels: int = 6
    shim_dma_channels: int = 2

    def __post_init__(self) -> None:
        if self.nb_cols_to_use <= 0:
            raise ValueError("nb_cols_to_use must be positive")


@dataclass(frozen=True)
class ConstraintOptStageConfig:
    transfer: TransferMilpConfig = field(default_factory=TransferMilpConfig)
    profiles: dict[str, CoreConstraintProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profiles:
            object.__setattr__(self, "profiles", default_core_profiles())
        elif len({p.type_name for p in self.profiles.values()}) != len(self.profiles):
            raise ValueError("Profile type_names must be unique")

    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> ConstraintOptStageConfig:
        """
        Temporary adapter to convert legacy kwargs into a typed config.
        """
        transfer_cfg = TransferMilpConfig(
            nb_cols_to_use=kwargs.get("nb_cols_to_use", 4),
        )
        profiles = kwargs.get("core_profiles", default_core_profiles())
        return cls(transfer=transfer_cfg, profiles=profiles)


def default_core_profiles() -> dict[str, CoreConstraintProfile]:
    return {
        "compute": CoreConstraintProfile(
            type_name="compute",
            roles={CoreRole.COMPUTE},
            capacity_operand=MemoryOperand("I2"),
        ),
        "memory": CoreConstraintProfile(
            type_name="memory",
            roles={CoreRole.CACHE},
            dma_limits=DmaBudget(mem_to_dram=6, dram_to_mem=6, shim_s2mm=2, shim_mm2s=2),
        ),
        "shim": CoreConstraintProfile(
            type_name="shim",
            roles={CoreRole.SHIM, CoreRole.IO},
        ),
    }


def pick_profile(profiles: dict[str, CoreConstraintProfile], core_type: str) -> CoreConstraintProfile:
    return profiles.get(core_type) or profiles.get("compute") or next(iter(profiles.values()))


def ensure_profile_registry(profiles: dict[str, CoreConstraintProfile] | None) -> dict[str, CoreConstraintProfile]:
    if profiles is None or not profiles:
        return default_core_profiles()
    return profiles
