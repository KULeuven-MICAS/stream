import json
import pathlib

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_addoption(parser):
    parser.addoption(
        "--keep-output",
        action="store_true",
        help="Keep generated outputs after tests finish",
    )


# --- Metrics Capture ---
# Bridge: module-level dict <- record_metric fixture <- CO test bodies
#         module-level dict -> pytest_terminal_summary -> metrics_current.json
# Note: incompatible with pytest-xdist worker isolation. If -n N is ever added to CI,
# each worker would have its own copy of _metrics_store and controller metrics would be lost.
# Acceptable for v1 (CI is single-process). KeyboardInterrupt
# (Ctrl-C) is not guaranteed to flush the file; only -x survival is required.

_metrics_store: dict[str, dict] = {}


@pytest.fixture
def record_metric(request):
    """Stash a CO metric for the regression guard.

    Injected into CO test signatures; call once per field after _assert_co_result.
    Each call is: record_metric("field_name", value_or_None). Keyed by the full
    pytest node ID so parametrized cells stay distinct.
    """
    node_id = request.node.nodeid

    def _record(key: str, value) -> None:
        _metrics_store.setdefault(node_id, {})[key] = value

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    """Conditionally write CO metrics to metrics_current.json at session end.

    Fires for all exit codes (pass, -x abort, failure). Writes ONLY when >=1
    metric was captured (non-empty store) -> runs with no CO tests never create or
    clobber the file. Uses a fixed absolute path at the repo root, derived from this
    conftest location (CWD-robust). Outer keys are full pytest node IDs, sorted.
    """
    if not _metrics_store:
        return
    out = pathlib.Path(__file__).parent.parent / "metrics_current.json"
    out.write_text(json.dumps(dict(sorted(_metrics_store.items())), indent=2, sort_keys=True))
