"""Installer for the optional AMD AIE code-generation toolchain (``stream-setup-aie``).

``pip install stream-dse`` gives you the base constraint-optimization pipeline. AIE MLIR
code generation additionally needs the AMD/Xilinx AIE toolchain, which cannot be expressed
as PyPI dependencies: the ``mlir_aie``/``llvm_aie`` wheels are platform-specific and hosted
on GitHub releases, ``xdsl-aie``/``snax-mlir`` are installed from git, and
``aie-python-extras`` must be built with a build-time environment variable. PyPI rejects all
such direct-reference dependencies, so this console script installs them into the active
environment instead.

Usage (after ``pip install stream-dse``)::

    stream-setup-aie            # install the toolchain into the current environment
    stream-setup-aie --dry-run  # print the steps without running them

Supported platform: Linux x86_64 with CPython 3.12 or 3.13 (matching the upstream Xilinx
wheels). After it completes, ``optimize_allocation_co_generic(..., enable_codegen=True)`` and
the AIE ``scripts/main_*.py`` entry points work without any further PYTHONPATH setup.
"""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

# --- Pinned AIE toolchain components -----------------------------------------------------
# Bump these together, and only after re-validating the AIE codegen path. Keeping the pins
# here (rather than in pyproject) is what lets stream-dse ship to PyPI with clean metadata.
_MLIR_AIE_WHEELS = {
    (3, 12): (
        "https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels/"
        "mlir_aie-0.0.1.2025070704+d7dc968-cp312-cp312-manylinux_2_35_x86_64.whl"
    ),
    (3, 13): (
        "https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels/"
        "mlir_aie-0.0.1.2025070704+d7dc968-cp313-cp313-manylinux_2_35_x86_64.whl"
    ),
}
_LLVM_AIE_WHEEL = (
    "https://github.com/Xilinx/llvm-aie/releases/download/nightly/"
    "llvm_aie-19.0.0.2025063001+6a9e0b4f-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
# xdsl-aie and snax-mlir pin xdsl to a git commit; install them with --no-deps so they do not
# clobber the released xdsl that stream-dse depends on (their runtime needs are already met).
_XDSL_AIE = "git+https://github.com/xdslproject/xdsl-aie.git@378c4c69c7f643ec31c6ef96c2fd830a0fb87244"
_SNAX_MLIR = "git+https://github.com/kuleuven-micas/snax-mlir.git@1c01c5d100df128c9fa01d3336ebea98e19b20cf"
# aie-python-extras provides aie.extras.context; it must be built with this prefix.
_AIE_PYTHON_EXTRAS = "git+https://github.com/makslevental/mlir-python-extras@f08db06"
_AIE_EXTRAS_BUILD_ENV = {"HOST_MLIR_PYTHON_PACKAGE_PREFIX": "aie"}

_PTH_FILENAME = "_stream_mlir_aie.pth"


def _check_platform() -> tuple[int, int]:
    """Validate the platform and return the (major, minor) Python version."""
    version = sys.version_info[:2]
    if sys.platform != "linux" or sysconfig.get_platform().split("-")[-1] not in ("x86_64", "amd64"):
        raise SystemExit(
            f"stream-setup-aie: the AIE toolchain is Linux x86_64 only "
            f"(got {sys.platform} / {sysconfig.get_platform()})."
        )
    if version not in _MLIR_AIE_WHEELS:
        supported = ", ".join(f"{a}.{b}" for a, b in sorted(_MLIR_AIE_WHEELS))
        raise SystemExit(
            f"stream-setup-aie: no mlir_aie wheel for Python {version[0]}.{version[1]}. "
            f"Upstream Xilinx publishes wheels for: {supported}."
        )
    return version


def _site_packages() -> Path:
    """Best-effort site-packages directory of the active interpreter."""
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    if paths:
        return Path(paths[0])
    return Path(sysconfig.get_paths()["purelib"])


def _steps(version: tuple[int, int]) -> list[tuple[str, list[str], dict[str, str]]]:
    """Return (description, pip-args, extra-env) for each install step."""
    pip = [sys.executable, "-m", "pip", "install"]
    return [
        ("mlir_aie + llvm_aie (Xilinx wheels)", [*pip, _MLIR_AIE_WHEELS[version], _LLVM_AIE_WHEEL], {}),
        ("xdsl-aie (git, no-deps)", [*pip, "--no-deps", _XDSL_AIE], {}),
        ("snax-mlir (git, no-deps)", [*pip, "--no-deps", _SNAX_MLIR], {}),
        ("aie-python-extras (custom build prefix)", [*pip, _AIE_PYTHON_EXTRAS], _AIE_EXTRAS_BUILD_ENV),
    ]


def _write_pth(dry_run: bool) -> Path | None:
    """Put mlir_aie's bundled python bindings on sys.path via a .pth file (no sourcing needed)."""
    site_packages = _site_packages()
    bindings = site_packages / "mlir_aie" / "python"
    pth = site_packages / _PTH_FILENAME
    if dry_run:
        print(f"  would write {pth} -> {bindings}")
        return pth
    pth.write_text(f"{bindings}\n")
    return pth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stream-setup-aie", description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without executing them.")
    args = parser.parse_args(argv)

    version = _check_platform()
    print(f"stream-setup-aie: installing the AIE toolchain for Python {version[0]}.{version[1]} (Linux x86_64)")

    for description, cmd, extra_env in _steps(version):
        print(f"\n==> {description}")
        if args.dry_run:
            env_prefix = " ".join(f"{k}={v}" for k, v in extra_env.items())
            print(f"  {env_prefix + ' ' if env_prefix else ''}{' '.join(cmd)}")
            continue
        env = {**os.environ, **extra_env}
        subprocess.run(cmd, check=True, env=env)

    print("\n==> mlir_aie python bindings")
    pth = _write_pth(args.dry_run)

    if args.dry_run:
        print("\nDry run complete — no changes made.")
    else:
        print(f"\nDone. mlir_aie bindings registered via {pth}.")
        print("AIE code generation (enable_codegen=True) is now available in this environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
