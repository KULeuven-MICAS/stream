"""Discovery of out-of-tree extensions, shared by every plugin registry.

Turns entry points into objects in one place, so every registry gets the same precedence, failure
behaviour and provenance. Loading is explicit: ``STREAM_OVERLAYS`` names the overlays a process may
load (a multi-tenant worker sets it per job). Precedence is declared: an overlay's ``OVERLAY_PRIORITY``
in the ``stream.overlays`` group decides which registration wins, the public distribution is priority 0,
and every conflict is logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

logger = logging.getLogger(__name__)

OVERLAY_GROUP = "stream.overlays"
OVERLAY_ALLOWLIST_ENV = "STREAM_OVERLAYS"
PUBLIC_DISTRIBUTION = "stream-dse"


@dataclass(frozen=True)
class LoadedPlugin:
    """One resolved entry point: what it registers, and where it came from."""

    name: str
    obj: Any
    distribution: str
    priority: int


def _distribution_of(ep: EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    return getattr(dist, "name", "") or ""


def _entry_points(group: str) -> list[EntryPoint]:
    try:
        return list(entry_points(group=group))
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        logger.debug("entry-point discovery failed for %r: %s", group, exc)
        return []


def overlay_allowlist() -> frozenset[str] | None:
    """Distributions this process may load overlays from (``STREAM_OVERLAYS``, comma-separated), or None
    when unrestricted. The public distribution is always allowed; an empty string means public only."""
    raw = os.environ.get(OVERLAY_ALLOWLIST_ENV)
    if raw is None:
        return None
    return frozenset({PUBLIC_DISTRIBUTION, *(part.strip() for part in raw.split(",") if part.strip())})


def overlay_priorities() -> dict[str, int]:
    """Priority per overlay distribution, from each overlay's ``OVERLAY_PRIORITY`` module constant."""
    priorities: dict[str, int] = {}
    for ep in _entry_points(OVERLAY_GROUP):
        distribution = _distribution_of(ep) or ep.name
        try:
            priorities[distribution] = int(getattr(ep.load(), "OVERLAY_PRIORITY", 1))
        except Exception as exc:  # pragma: no cover - a broken overlay must not break discovery
            logger.warning("overlay %r declared itself but could not be loaded: %s", ep.name, exc)
    return priorities


def loaded_overlays() -> tuple[str, ...]:
    """Overlay distributions visible to this process, after the allowlist. Provenance for a run: two
    results are only comparable when they were produced with the same overlays."""
    allow = overlay_allowlist()
    names = (d for d in overlay_priorities() if allow is None or d in allow)
    return tuple(sorted(names))


def load_group(group: str, allow: frozenset[str] | None | object = ...) -> list[LoadedPlugin]:
    """Entry points in ``group``, lowest priority first so the last registration wins. Conflicts are
    logged; a plugin that fails to load is skipped, never raised. ``allow`` defaults to
    :func:`overlay_allowlist`; pass an explicit set to override it for one call."""
    allowed = overlay_allowlist() if allow is ... else allow
    priorities = overlay_priorities()

    loaded: list[LoadedPlugin] = []
    for ep in _entry_points(group):
        distribution = _distribution_of(ep)
        if allowed is not None and distribution and distribution not in allowed:
            logger.debug("skipping %r from %r: not in %s", ep.name, distribution, OVERLAY_ALLOWLIST_ENV)
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # pragma: no cover - a broken plugin must not break the registry
            logger.warning("skipping %s plugin %r: %s", group, ep.name, exc)
            continue
        loaded.append(LoadedPlugin(ep.name, obj, distribution, priorities.get(distribution, 0)))

    loaded.sort(key=lambda p: (p.priority, p.distribution, p.name))
    _warn_on_conflicts(group, loaded)
    return loaded


def _warn_on_conflicts(group: str, loaded: list[LoadedPlugin]) -> None:
    by_name: dict[str, list[LoadedPlugin]] = {}
    for plugin in loaded:
        by_name.setdefault(plugin.name, []).append(plugin)
    for name, plugins in by_name.items():
        if len(plugins) > 1:
            winner = plugins[-1]
            others = ", ".join(f"{p.distribution}(priority {p.priority})" for p in plugins[:-1])
            logger.warning(
                "%s: %r registered by several distributions; %s(priority %d) wins over %s",
                group,
                name,
                winner.distribution,
                winner.priority,
                others,
            )
