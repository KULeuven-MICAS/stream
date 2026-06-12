"""Structural assertions for `.github/workflows/ci.yml` (Phase 41, CI-01..CI-04).

These 10 fast tests verify that the CI workflow is wired correctly for the v1.10
metrics-regression guard.  They are purely structural — they parse YAML and grep
raw text; they do not run tests, call the solver, or touch the network.

The behavioral gate (sticky comment on a real PR; advisory-only; fork-skip) is a
HUMAN-UAT item that cannot be automated locally.
"""

import pathlib

import yaml

# Resolve ci.yml relative to this file so the test works regardless of CWD.
_CI_YML = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def _load():
    """Return (parsed_dict, raw_text) for ci.yml."""
    raw = _CI_YML.read_text()
    parsed = yaml.safe_load(raw)
    return parsed, raw


# ---------------------------------------------------------------------------
# 1. Fork-guard present (CI-03)
# ---------------------------------------------------------------------------


def test_fork_guard_present():
    """metrics-comment job if: condition must include the fork-guard expression (CI-03)."""
    _, raw = _load()
    assert "github.event.pull_request.head.repo.full_name == github.repository" in raw, (
        "Fork-guard expression not found in ci.yml — CI-03 violated"
    )


# ---------------------------------------------------------------------------
# 2. Upload step has if: always() (D-02)
# ---------------------------------------------------------------------------


def test_upload_if_always():
    """The upload-artifact step in the tests job must carry 'if: always()' (D-02)."""
    _, raw = _load()
    # The raw text must contain both the upload-artifact action and an if: always() nearby.
    # We check both markers are present in the file (Task 1 places them adjacent).
    assert "upload-artifact@v4" in raw, "upload-artifact@v4 step not found in ci.yml"
    assert "if: always()" in raw, "'if: always()' not found in ci.yml (D-02 violated)"


# ---------------------------------------------------------------------------
# 3. Upload step has continue-on-error: true (D-02)
# ---------------------------------------------------------------------------


def test_upload_continue_on_error():
    """The upload-artifact step must carry continue-on-error: true (D-02)."""
    parsed, _ = _load()
    tests_steps = parsed["jobs"]["tests"]["steps"]
    upload_steps = [s for s in tests_steps if isinstance(s.get("uses"), str) and "upload-artifact" in s["uses"]]
    assert upload_steps, "No upload-artifact step found in tests job"
    upload_step = upload_steps[0]
    assert upload_step.get("continue-on-error") is True, (
        f"upload-artifact step missing continue-on-error: true (D-02); got: {upload_step.get('continue-on-error')!r}"
    )


# ---------------------------------------------------------------------------
# 4. Top-level permissions is contents:read only — no pull-requests at top level
# ---------------------------------------------------------------------------


def test_top_level_permissions_read_only():
    """Top-level permissions must be {{contents: read}} only — pull-requests must NOT appear there (scope isolation)."""
    parsed, _ = _load()
    top_perms = parsed.get("permissions")
    assert top_perms == {"contents": "read"}, (
        f"Top-level permissions should be exactly {{contents: read}}, got: {top_perms!r}"
    )
    assert "pull-requests" not in (top_perms or {}), (
        "pull-requests: write must NOT be a top-level permission — only scoped to metrics-comment job"
    )


# ---------------------------------------------------------------------------
# 5. metrics-comment job permissions include pull-requests: write (D-03)
# ---------------------------------------------------------------------------


def test_comment_job_has_pr_write():
    """metrics-comment job permissions must include pull-requests: write (D-03)."""
    parsed, _ = _load()
    job_perms = parsed["jobs"]["metrics-comment"]["permissions"]
    assert job_perms.get("pull-requests") == "write", (
        f"metrics-comment job missing 'pull-requests: write'; got permissions: {job_perms!r}"
    )


# ---------------------------------------------------------------------------
# 6. metrics-comment job permissions include contents: read (D-03 — per-job perms fully replace)
# ---------------------------------------------------------------------------


def test_comment_job_has_contents_read():
    """metrics-comment job must explicitly declare contents: read (per-job perms fully replace top-level, D-03)."""
    parsed, _ = _load()
    job_perms = parsed["jobs"]["metrics-comment"]["permissions"]
    assert job_perms.get("contents") == "read", (
        f"metrics-comment job missing explicit 'contents: read'; got permissions: {job_perms!r}"
    )


# ---------------------------------------------------------------------------
# 7. Concurrency group keyed by PR number (D-07)
# ---------------------------------------------------------------------------


def test_concurrency_group_keyed_by_pr():
    """metrics-comment job concurrency group must contain github.event.pull_request.number (D-07)."""
    parsed, _ = _load()
    concurrency = parsed["jobs"]["metrics-comment"].get("concurrency", {})
    group = concurrency.get("group", "")
    assert "github.event.pull_request.number" in group, (
        f"Concurrency group must be keyed by PR number (D-07); got: {group!r}"
    )


# ---------------------------------------------------------------------------
# 8. find-comment body-includes contains the sticky-comment marker (CI-01)
# ---------------------------------------------------------------------------


def test_find_comment_marker():
    """find-comment body-includes must contain stream-aie-metrics-regression-guard-v1 (CI-01 / Pitfall 6)."""
    _, raw = _load()
    assert "stream-aie-metrics-regression-guard-v1" in raw, (
        "Sticky-comment marker 'stream-aie-metrics-regression-guard-v1' not found in ci.yml — "
        "find-comment will fail to locate existing comments (duplicate-comment bug, CI-01)"
    )


# ---------------------------------------------------------------------------
# 9. Exactly one pytest invocation in ci.yml (CI-04 — matrix runs once)
# ---------------------------------------------------------------------------


def test_single_pytest_invocation():
    """ci.yml must contain exactly one 'pytest' substring — the matrix runs exactly once (CI-04)."""
    _, raw = _load()
    count = raw.count("pytest")
    assert count == 1, f"Expected exactly 1 'pytest' invocation in ci.yml (CI-04 matrix-once), found {count}"


# ---------------------------------------------------------------------------
# 10. metrics-comment job needs: [tests] (D-03)
# ---------------------------------------------------------------------------


def test_comment_job_needs_tests():
    """metrics-comment job must declare needs: [tests] so it runs after the tests job (D-03)."""
    parsed, _ = _load()
    needs = parsed["jobs"]["metrics-comment"].get("needs", [])
    # needs may be a list or a string — normalise
    if isinstance(needs, str):
        needs = [needs]
    assert needs == ["tests"], f"metrics-comment job needs must be ['tests'], got: {needs!r}"
