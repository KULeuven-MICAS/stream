from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransferMilpConfig:
    nb_cols_to_use: int = 4

    def __post_init__(self) -> None:
        if self.nb_cols_to_use <= 0:
            raise ValueError("nb_cols_to_use must be positive")


@dataclass(frozen=True)
class ConstraintOptStageConfig:
    transfer: TransferMilpConfig = field(default_factory=TransferMilpConfig)

    @classmethod
    def from_kwargs(cls, **kwargs) -> ConstraintOptStageConfig:
        """Build the config from the stage context, reading `nb_cols_to_use`."""
        transfer_cfg = TransferMilpConfig(
            nb_cols_to_use=kwargs.get("nb_cols_to_use", 4),
        )
        return cls(transfer=transfer_cfg)
