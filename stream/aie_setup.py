"""Installer for the optional AMD AIE code-generation toolchain (``stream-setup-aie``).

``pip install stream-dse`` gives you the base constraint-optimization pipeline. AIE MLIR
code generation additionally needs the AMD/Xilinx AIE toolchain, which cannot be expressed
as PyPI dependencies: the ``mlir_aie``/``llvm-aie`` wheels are platform-specific and hosted
on GitHub releases, and ``xdsl-aie``/``snax-mlir`` are installed from git. PyPI rejects all
such direct-reference dependencies, so this console script installs them into the active
environment instead.

Usage (after ``pip install stream-dse``)::

    stream-setup-aie            # install the toolchain into the current environment
    stream-setup-aie --dry-run  # print the steps without running them

Supported platform: Linux x86_64 with CPython 3.10-3.12 or 3.14 (matching the pinned upstream
Xilinx wheels). After it completes, ``optimize_allocation_co_generic(..., enable_codegen=True)``
and the AIE ``scripts/main_*.py`` entry points work without any further PYTHONPATH setup.
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
# These mirror amd/iron's `devel` requirements.txt (the toolchain that consumes the MLIR we
# emit, and the pickiest pin in the stack). Bump them in lockstep with IRON's pins, and only
# after re-validating the AIE codegen path. We install mlir_aie / llvm-aie with pip from the
# GitHub "expanded_assets" index pages (the same mechanism IRON uses) rather than hardcoded
# wheel URLs: the rolling release tags rotate old builds out within days, so a hardcoded URL
# rots. We list both the rolling `nightly` and the dated archival llvm-aie tag, so the pinned
# build still resolves once it has left the rolling window. pip auto-selects the wheel matching
# the active interpreter/platform. Keeping the pins here (not in pyproject) is what lets
# stream-dse ship to PyPI with clean metadata.
_MLIR_AIE_PIN = "mlir_aie==0.0.1.2026033104+e4f35d6"  # amd/iron devel
_LLVM_AIE_PIN = "llvm-aie==21.0.0.2026051101+adc9df1a"  # amd/iron devel
_AIE_WHEEL_INDICES = [
    "--extra-index-url",
    "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3",
    "--extra-index-url",
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly",
    "--extra-index-url",
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly-20240501-20260527",
]
# CPython versions for which the pinned mlir_aie build publishes wheels on latest-wheels-3.
_SUPPORTED_PYTHONS = {(3, 10), (3, 11), (3, 12), (3, 14)}
# xdsl-aie and snax-mlir pin xdsl to a git commit; install them with --no-deps so they do not
# clobber the released xdsl that stream-dse depends on (their runtime needs are already met).
_XDSL_AIE = "git+https://github.com/xdslproject/xdsl-aie.git@378c4c69c7f643ec31c6ef96c2fd830a0fb87244"
_SNAX_MLIR = "git+https://github.com/kuleuven-micas/snax-mlir.git@1c01c5d100df128c9fa01d3336ebea98e19b20cf"
# Note: aie.extras.context / aie.utils.trace (the optional tracing helpers) are now bundled in
# the mlir_aie wheel itself, reachable once the .pth below puts mlir_aie/python on sys.path.
# The old separate makslevental/mlir-python-extras pin is therefore dropped: it is redundant and
# no longer builds (its eudsl-llvmpy build dependency is unpublished).

_PTH_FILENAME = "_stream_mlir_aie.pth"


def _check_platform() -> tuple[int, int]:
    """Validate the platform and return the (major, minor) Python version."""
    version = sys.version_info[:2]
    if sys.platform != "linux" or sysconfig.get_platform().split("-")[-1] not in ("x86_64", "amd64"):
        raise SystemExit(
            f"stream-setup-aie: the AIE toolchain is Linux x86_64 only "
            f"(got {sys.platform} / {sysconfig.get_platform()})."
        )
    if version not in _SUPPORTED_PYTHONS:
        supported = ", ".join(f"{a}.{b}" for a, b in sorted(_SUPPORTED_PYTHONS))
        raise SystemExit(
            f"stream-setup-aie: no mlir_aie wheel for Python {version[0]}.{version[1]}. "
            f"The pinned build publishes wheels for: {supported}."
        )
    return version


def _site_packages() -> Path:
    """Best-effort site-packages directory of the active interpreter."""
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    if paths:
        return Path(paths[0])
    return Path(sysconfig.get_paths()["purelib"])


def _steps() -> list[tuple[str, list[str], dict[str, str]]]:
    """Return (description, pip-args, extra-env) for each install step."""
    pip = [sys.executable, "-m", "pip", "install"]
    return [
        (
            "mlir_aie + llvm-aie (Xilinx wheels via GitHub release index)",
            [*pip, *_AIE_WHEEL_INDICES, _MLIR_AIE_PIN, _LLVM_AIE_PIN],
            {},
        ),
        ("xdsl-aie (git, no-deps)", [*pip, "--no-deps", _XDSL_AIE], {}),
        ("snax-mlir (git, no-deps)", [*pip, "--no-deps", _SNAX_MLIR], {}),
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

    for description, cmd, extra_env in _steps():
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
