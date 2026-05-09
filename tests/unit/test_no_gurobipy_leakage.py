"""Verify no gurobipy imports leaked outside the solver backend.

After Phase 1 refactoring, the only permitted gurobipy imports are:
1. stream/opt/solver/solver.py (the backend implementation)
2. transfer_and_tensor_allocation.py: `from gurobipy import GRB` (callback constants, per D-04)
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CO_DIR = REPO_ROOT / "stream" / "opt" / "allocation" / "constraint_optimization"

# Files that MUST NOT have any gurobipy imports
CLEAN_FILES = [
    CO_DIR / "allocation.py",
    CO_DIR / "context.py",
    CO_DIR / "utils.py",
    REPO_ROOT / "stream" / "api.py",
]

# File with ONE permitted import: `from gurobipy import GRB`
PERMITTED_EXCEPTION = CO_DIR / "transfer_and_tensor_allocation.py"


def _get_gurobipy_imports(filepath: Path) -> list[str]:
    """Parse file AST and return all gurobipy import statements."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "gurobipy" in alias.name:
                    imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and "gurobipy" in node.module:
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {node.module} import {names}")
    return imports


def test_no_gurobipy_in_co_files():
    """All CO files except solver.py must be gurobipy-free."""
    violations = {}
    for filepath in CLEAN_FILES:
        assert filepath.exists(), f"Expected file not found: {filepath}"
        imports = _get_gurobipy_imports(filepath)
        if imports:
            violations[str(filepath.relative_to(REPO_ROOT))] = imports

    assert not violations, "Gurobipy imports found outside solver backend:\n" + "\n".join(
        f"  {path}: {imps}" for path, imps in violations.items()
    )


def test_tta_only_imports_grb_constants():
    """TTA may import GRB (for callback) but NOT gp.Model or gp.Var."""
    assert PERMITTED_EXCEPTION.exists()
    imports = _get_gurobipy_imports(PERMITTED_EXCEPTION)
    # Only "from gurobipy import GRB" is allowed
    for imp in imports:
        assert imp == "from gurobipy import GRB", (
            f"Unexpected gurobipy import in TTA: {imp!r}. "
            f"Only 'from gurobipy import GRB' is permitted (for callback constants, per D-04)."
        )
