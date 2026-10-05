import json
import pathlib
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel

TWO_CONV = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)


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


def _solve_model(
    hardware: str,
    workload: Any,
    output_path: str,
    mapping: Any = None,
    options: SolveOptions | None = None,
    *,
    families_available: Mapping[str, Callable[..., Any]] | None = None,
    hook: str = "solve",
) -> AllocationModel:
    """Run ``evaluate_mapping`` and return the first allocation model that reached its ``hook`` method (``solve`` or
    ``_build_model``); ``families_available`` replaces the entry-point families."""
    captured: list[AllocationModel] = []
    original = getattr(AllocationModel, hook)

    def capture(self: AllocationModel, *args: Any, **kwargs: Any) -> Any:
        captured.append(self)
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        if families_available is not None:
            patch.setattr(families, "available_families", lambda: dict(families_available))
        patch.setattr(AllocationModel, hook, capture)
        evaluate_mapping(hardware, workload, output_path, mapping, options)
    return captured[0]


def _model_size(alloc: AllocationModel) -> tuple[int, int]:
    """(variables, linear constraints) of the allocation model, as built."""
    raw = alloc.model._model  # type: ignore[attr-defined]
    return sum(1 for _ in raw.variables()), sum(1 for _ in raw.linear_constraints())


def _interval(alloc: AllocationModel) -> float:
    """The solved initiation interval: the iteration minus the overlap with the next one."""
    value, q = alloc.model.value, alloc.quantities
    return value(q.get("iteration").expr) - value(q.get("overlap").expr)


@pytest.fixture(scope="session")
def two_conv() -> TwoConvWorkloadConfig:
    return TWO_CONV


@pytest.fixture(scope="session")
def solved_model() -> Callable[..., AllocationModel]:
    """:func:`_solve_model`; session scoped so module-scoped fixtures can solve once."""
    return _solve_model


@pytest.fixture(scope="session")
def model_size() -> Callable[[AllocationModel], tuple[int, int]]:
    return _model_size


@pytest.fixture(scope="session")
def interval() -> Callable[[AllocationModel], float]:
    return _interval
