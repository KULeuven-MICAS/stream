from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageContext:
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> StageContext:
        return cls(data=dict(kwargs))

    def require_fields(self, fields: Iterable[str], stage_name: str) -> None:
        missing = [f for f in fields if f not in self.data or self.data[f] is None]
        if missing:
            raise ValueError(f"{stage_name} missing required context fields: {', '.join(missing)}")

    def require_value(self, key: str, stage_name: str) -> Any:
        if key not in self.data or self.data[key] is None:
            raise ValueError(f"{stage_name} missing required context field: {key}")
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, **kwargs: Any) -> None:
        self.data.update(kwargs)
