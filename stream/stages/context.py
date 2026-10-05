from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_RUNNING: ContextVar[Any] = ContextVar("stream_running_stage", default=None)


class StageContractError(Exception):
    """A stage reads or writes a context field its contract does not declare, or reads one nothing wrote."""


@contextmanager
def running(stage: type) -> Iterator[None]:
    """Check every context access in this block against the contract of ``stage``."""
    token = _RUNNING.set(stage)
    try:
        yield
    finally:
        _RUNNING.reset(token)


@dataclass
class StageContext:
    """The fields the stages of a pipeline hand each other. While a stage runs, it can read and write only the
    fields its contract declares (see :class:`~stream.stages.stage.Stage`); outside a stage every field is open."""

    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> StageContext:
        return cls(data=dict(kwargs))

    def get(self, key: str, default: Any = None) -> Any:
        stage = _RUNNING.get()
        if stage is not None and key not in stage.readable:
            raise StageContractError(f"{stage.__name__} reads {key!r}, which its contract does not declare")
        return self.data.get(key, default)

    def set(self, **kwargs: Any) -> None:
        self._check_writes(kwargs)
        self.data.update(kwargs)

    def pop(self, key: str, default: Any = None) -> Any:
        self._check_writes((key,))
        return self.data.pop(key, default)

    @staticmethod
    def _check_writes(keys: Any) -> None:
        stage = _RUNNING.get()
        if stage is not None and (undeclared := [key for key in keys if key not in stage.writable]):
            raise StageContractError(f"{stage.__name__} writes {undeclared}, which its contract does not declare")
