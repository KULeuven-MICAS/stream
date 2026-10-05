"""Every key of every YAML file a solve writes is documented in docs/source/outputs.md, and every documented key is
written."""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.solver import GurobiBackend

TPU = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
AIE = "stream/inputs/aie/hardware/whole_array_strix.yaml"
HEADING = re.compile(r"^### `(.+\.yaml)`$")
ROW = re.compile(r"^\| `([^`]+)` \|")
GUROBI_ONLY = {"allocation/reports/optimization_trace.yaml"}


def _documented() -> dict[str, set[str]]:
    """Per YAML file, relative to its group's folder, the key paths outputs.md lists for it."""
    tables: dict[str, set[str]] = {}
    current = None
    for line in Path("docs/source/outputs.md").read_text().splitlines():
        if heading := HEADING.match(line):
            current = tables.setdefault(heading.group(1), set())
        elif line.startswith("#"):
            current = None
        elif current is not None and (row := ROW.match(line)):
            current.add(row.group(1))
    return tables


def _paths(node: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from _paths(value, path)
    elif isinstance(node, list):
        for item in node:
            yield from _paths(item, f"{prefix}[]")


def _pattern(documented: str) -> re.Pattern[str]:
    return re.compile(re.sub(r"<\w+>", r"[^.\\[\\]]+", re.escape(documented)))


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory, two_conv: TwoConvWorkloadConfig) -> tuple[Path, str]:
    """A ZigZag and an AIE solve with their artifacts, by Gurobi where it is licensed."""
    try:
        GurobiBackend.check_license()
        backend = "gurobi"
    except (AttributeError, ValueError):
        backend = "ortools_gscip"
    root = tmp_path_factory.mktemp("yaml_outputs")
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)
    evaluate_mapping(TPU, workload, str(root / "tpu"), mapping, SolveOptions(backend=backend))
    workload = make_gemm_workload(128, 128, 128, "bf16", "bf16")
    mapping = make_gemm_mapping(128, 128, 128, 64, 64, 64, nb_rows_to_use=2, nb_cols_to_use=2)
    evaluate_mapping(AIE, workload, str(root / "aie"), mapping, SolveOptions(backend=backend, nb_cols_to_use=2))
    return root, backend


def test_every_yaml_file_a_solve_writes_is_documented(solved: tuple[Path, str]) -> None:
    root, backend = solved
    written = {str(file.relative_to(group)) for group in root.glob("*/group_*") for file in group.rglob("*.yaml")}
    expected = set(_documented()) - (set() if backend == "gurobi" else GUROBI_ONLY)
    assert written == expected


@pytest.mark.parametrize("name", sorted(_documented()))
def test_a_yaml_file_holds_exactly_the_documented_keys(solved: tuple[Path, str], name: str) -> None:
    root, backend = solved
    if name in GUROBI_ONLY and backend != "gurobi":
        pytest.skip("only a Gurobi solve writes it")
    written = {path for file in root.glob(f"*/group_*/{name}") for path in _paths(yaml.safe_load(file.read_text()))}
    documented = _documented()[name]
    patterns = {key: _pattern(key) for key in documented}
    undocumented = {path for path in written if not any(p.fullmatch(path) for p in patterns.values())}
    absent = {key for key, p in patterns.items() if not any(p.fullmatch(path) for path in written)}
    assert not undocumented, f"{name} holds keys outputs.md does not document: {sorted(undocumented)}"
    assert not absent, f"outputs.md documents keys {name} does not hold: {sorted(absent)}"
