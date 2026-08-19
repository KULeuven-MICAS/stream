"""The plugin registry: allowlist, declared precedence, and broken-overlay containment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stream import plugins
from stream.plugins import PUBLIC_DISTRIBUTION, load_group, loaded_overlays, overlay_allowlist


class _EntryPoint:
    """Enough of importlib.metadata.EntryPoint for the registry: a name, a distribution, a load()."""

    def __init__(self, name: str, obj: object, distribution: str, *, broken: bool = False):
        self.name = name
        self.dist = SimpleNamespace(name=distribution)
        self._obj = obj
        self._broken = broken

    def load(self):
        if self._broken:
            raise RuntimeError("this overlay is broken")
        return self._obj


@pytest.fixture
def entry_points(monkeypatch):
    """Install synthetic entry points per group."""
    groups: dict[str, list[_EntryPoint]] = {}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: groups.get(group, []))
    return groups


def _overlay(priority: int) -> SimpleNamespace:
    return SimpleNamespace(OVERLAY_PRIORITY=priority)


# --------------------------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------------------------- #


def test_unset_allowlist_is_unrestricted(monkeypatch):
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    assert overlay_allowlist() is None


def test_empty_allowlist_restricts_to_public(monkeypatch):
    """An empty value is not 'unset' -- it is a deliberate 'public code only'."""
    monkeypatch.setenv(plugins.OVERLAY_ALLOWLIST_ENV, "")
    assert overlay_allowlist() == frozenset({PUBLIC_DISTRIBUTION})


def test_allowlist_always_includes_the_public_distribution(monkeypatch):
    monkeypatch.setenv(plugins.OVERLAY_ALLOWLIST_ENV, "vendor-overlay-acme")
    assert overlay_allowlist() == frozenset({PUBLIC_DISTRIBUTION, "vendor-overlay-acme"})


def test_load_group_excludes_overlays_outside_the_allowlist(monkeypatch, entry_points):
    """Isolation: an overlay outside the allowlist is excluded even when installed in the same env."""
    monkeypatch.setenv(plugins.OVERLAY_ALLOWLIST_ENV, "vendor-overlay-acme")
    entry_points["stream.decompositions"] = [
        _EntryPoint("PublicOp", "public", PUBLIC_DISTRIBUTION),
        _EntryPoint("AcmeOp", "acme", "vendor-overlay-acme"),
        _EntryPoint("GlobexOp", "globex", "vendor-overlay-globex"),
    ]
    assert [p.name for p in load_group("stream.decompositions")] == ["PublicOp", "AcmeOp"]


def test_explicit_allow_overrides_the_environment(monkeypatch, entry_points):
    monkeypatch.setenv(plugins.OVERLAY_ALLOWLIST_ENV, "vendor-overlay-acme")
    entry_points["stream.decompositions"] = [_EntryPoint("GlobexOp", "globex", "vendor-overlay-globex")]
    assert [p.name for p in load_group("stream.decompositions", allow=None)] == ["GlobexOp"]


# --------------------------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------------------------- #


def test_higher_priority_overlay_is_registered_last(monkeypatch, entry_points):
    """Higher-priority overlays are returned last (last wins): overlay outranks built-in."""
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    entry_points[plugins.OVERLAY_GROUP] = [
        _EntryPoint("shared", _overlay(10), "vendor-overlay"),
        _EntryPoint("acme", _overlay(20), "vendor-overlay-acme"),
    ]
    entry_points["stream.onnx_parsers"] = [
        _EntryPoint("Softmax", "acme", "vendor-overlay-acme"),
        _EntryPoint("Softmax", "builtin", PUBLIC_DISTRIBUTION),
        _EntryPoint("Softmax", "shared", "vendor-overlay"),
    ]
    assert [p.obj for p in load_group("stream.onnx_parsers")] == ["builtin", "shared", "acme"]


def test_ordering_is_stable_without_declared_priorities(monkeypatch, entry_points):
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    entry_points["stream.decompositions"] = [
        _EntryPoint("B", "b", "z-dist"),
        _EntryPoint("A", "a", "a-dist"),
    ]
    assert [p.distribution for p in load_group("stream.decompositions")] == ["a-dist", "z-dist"]


def test_conflicting_registrations_are_logged(monkeypatch, entry_points, caplog):
    """A silent override is how one overlay quietly changes another customer's numbers."""
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    entry_points[plugins.OVERLAY_GROUP] = [_EntryPoint("acme", _overlay(20), "vendor-overlay-acme")]
    entry_points["stream.onnx_parsers"] = [
        _EntryPoint("Softmax", "builtin", PUBLIC_DISTRIBUTION),
        _EntryPoint("Softmax", "acme", "vendor-overlay-acme"),
    ]
    with caplog.at_level("WARNING"):
        load_group("stream.onnx_parsers")
    assert "Softmax" in caplog.text
    assert "vendor-overlay-acme" in caplog.text


# --------------------------------------------------------------------------------------------- #
# Failure containment and provenance
# --------------------------------------------------------------------------------------------- #


def test_a_broken_plugin_is_skipped_not_raised(monkeypatch, entry_points):
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    entry_points["stream.decompositions"] = [
        _EntryPoint("Broken", None, "vendor-overlay", broken=True),
        _EntryPoint("Fine", "fine", "vendor-overlay"),
    ]
    assert [p.name for p in load_group("stream.decompositions")] == ["Fine"]


def test_a_broken_overlay_declaration_does_not_break_discovery(monkeypatch, entry_points):
    monkeypatch.delenv(plugins.OVERLAY_ALLOWLIST_ENV, raising=False)
    entry_points[plugins.OVERLAY_GROUP] = [_EntryPoint("bad", None, "vendor-overlay-bad", broken=True)]
    entry_points["stream.decompositions"] = [_EntryPoint("Op", "op", PUBLIC_DISTRIBUTION)]
    assert [p.name for p in load_group("stream.decompositions")] == ["Op"]


def test_loaded_overlays_reports_provenance(monkeypatch, entry_points):
    """Two results are only comparable when they were produced with the same overlays."""
    monkeypatch.setenv(plugins.OVERLAY_ALLOWLIST_ENV, "vendor-overlay-acme")
    entry_points[plugins.OVERLAY_GROUP] = [
        _EntryPoint("acme", _overlay(20), "vendor-overlay-acme"),
        _EntryPoint("globex", _overlay(20), "vendor-overlay-globex"),
    ]
    assert loaded_overlays() == ("vendor-overlay-acme",)
