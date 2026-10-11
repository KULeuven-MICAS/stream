"""Every solver backend reaches the default backend's objective on the AIE GEMM and SwiGLU allocations.

The Gurobi backends run only where gurobipy is installed; the nightly job leaves them out, as it has no license."""

import importlib.util
import os
import tempfile
from functools import cache

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

ACCELERATOR = os.path.join(os.path.dirname(__file__), "../../stream/inputs/aie/hardware/whole_array_strix.yaml")
COLUMNS = 8
REL_TOL = 0.01
_GUROBI = pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")


def _gemm() -> tuple[str, str]:
    M, K, N = 256, 8192, 2048  # noqa: N806
    workload = make_gemm_workload(M, K, N, "bf16", "bf16")
    return workload, make_gemm_mapping(M, K, N, 32, 32, 32, nb_rows_to_use=4, nb_cols_to_use=COLUMNS)


def _swiglu() -> tuple[str, str]:
    seq_len, embedding, hidden = 256, 512, 2048
    workload = make_swiglu_workload(seq_len, embedding, hidden, "bf16", "bf16")
    return workload, make_swiglu_mapping(seq_len, embedding, hidden, True, 32, 32, 64)


INSTANCES = {"gemm": _gemm, "swiglu": _swiglu}


@cache
def _objective(instance: str, backend: str) -> float:
    """The latency the allocation of ``instance`` minimizes, solved by ``backend``."""
    workload, mapping = INSTANCES[instance]()
    with tempfile.TemporaryDirectory() as output:
        options = SolveOptions(backend=backend, nb_cols_to_use=COLUMNS, artifacts=False)
        context = evaluate_mapping(ACCELERATOR, workload, output, mapping, options).context
    return float(context.get("allocation").solution.latency.total)


@pytest.mark.slow
@pytest.mark.parametrize("instance", INSTANCES)
@pytest.mark.parametrize(
    "backend",
    ["ortools_highs", pytest.param("gurobi", marks=_GUROBI), pytest.param("ortools_gurobi", marks=_GUROBI)],
)
def test_a_backend_reaches_the_default_backends_objective(instance: str, backend: str):
    reference = _objective(instance, SolveOptions.backend)
    assert _objective(instance, backend) == pytest.approx(reference, rel=REL_TOL)
