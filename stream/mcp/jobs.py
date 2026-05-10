"""stream.mcp.jobs — Job registry dataclass and content-addressed experiment ID helper."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerState:
    """In-process job registry. Lost on server restart — completed jobs
    are recoverable from ctx.pickle via poll_optimization.

    Each job value has shape: {"status": str, "result": dict | None, "error": str | None}
    Valid status values: "pending", "running", "complete", "failed"
    """

    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)


def make_experiment_id(
    hardware: str,
    workload: str,
    mapping: str,
    backend: str,
    constraint_dict: dict[str, bool],
) -> str:
    """Content-addressed experiment ID per D-07.

    SHA-256 of file CONTENTS (not paths) + backend string + sorted JSON of constraint_dict,
    truncated to 12 lowercase hex characters.

    Args:
        hardware: Path to hardware YAML file.
        workload: Path to workload ONNX file.
        mapping: Path to mapping YAML file.
        backend: Solver backend string (e.g. "ortools_gscip").
        constraint_dict: Dict of constraint bool flags (e.g. {"memory_capacity": True, ...}).

    Returns:
        12-character lowercase hex string derived from SHA-256 hash of all inputs.
    """
    h = hashlib.sha256()
    for path in (hardware, workload, mapping):
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(backend.encode())
    h.update(json.dumps(constraint_dict, sort_keys=True).encode())
    return h.hexdigest()[:12]
