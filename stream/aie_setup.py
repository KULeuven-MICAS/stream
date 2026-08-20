"""Installer for stream-dse's optional AIE code-generation dependencies (``stream-setup-aie``).

Codegen emits text MLIR via xDSL and needs two dialect packages that PyPI forbids as
dependencies (they pin ``xdsl`` to a git commit): ``xdsl-aie`` and ``snax-mlir``. This console
script installs them into the active environment::

    stream-setup-aie                  # install the codegen dialects
    stream-setup-aie --dry-run        # print the steps without running them
    stream-setup-aie --with-mlir-aie  # additionally install the mlir_aie/llvm-aie wheels

``mlir_aie``/``llvm-aie`` are NOT installed by default: codegen never imports the ``aie``
bindings, and the host that compiles the emitted MLIR (e.g. amd/iron) already pins its own,
newer wheels -- reinstalling here would downgrade and break it. ``--with-mlir-aie`` is the
opt-in for a standalone toolchain, and is itself a no-op when a ``mlir_aie`` is already present.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

# Installed --no-deps: they pin xdsl to a git commit that would otherwise clobber the released
# xdsl that stream-dse depends on.
# TODO: repoint at xdslproject/xdsl-aie once PR 28 merges; this fork commit is a stand-in.
_XDSL_AIE = "git+https://github.com/KULeuven-MICAS/xdsl-aie.git@2674ab272ce4f9597ff8afd77772527f2d887ab4"
_SNAX_MLIR = "git+https://github.com/kuleuven-micas/snax-mlir.git@1c01c5d100df128c9fa01d3336ebea98e19b20cf"

# Opt-in only (--with-mlir-aie). Codegen emits the v1.4.0 buffer descriptor form, which
# earlier mlir-aie parses by dropping the attributes it does not know rather than failing,
# so a standalone toolchain has to be v1.4.0 or newer. This llvm-aie is the build amd/iron
# pairs with it. Several llvm-aie indices are listed because a pinned build ages out of the
# rolling `nightly` window into the dated archival tag.
_MLIR_AIE_PIN = "mlir_aie==1.4.0"
_LLVM_AIE_PIN = "llvm-aie==21.0.0.2026062301+cb664e8c"
_AIE_WHEEL_INDICES = [
    "--extra-index-url",
    "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.0",
    "--extra-index-url",
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly",
    "--extra-index-url",
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly-20240501-20260527",
]
_SUPPORTED_PYTHONS = {(3, 10), (3, 11), (3, 12), (3, 14)}

_PTH_FILENAME = "_stream_mlir_aie.pth"


def _check_platform() -> None:
    version = sys.version_info[:2]
    if sys.platform != "linux" or sysconfig.get_platform().split("-")[-1] not in ("x86_64", "amd64"):
        raise SystemExit(
            f"stream-setup-aie: --with-mlir-aie is Linux x86_64 only (got {sys.platform} / {sysconfig.get_platform()})."
        )
    if version not in _SUPPORTED_PYTHONS:
        supported = ", ".join(f"{a}.{b}" for a, b in sorted(_SUPPORTED_PYTHONS))
        raise SystemExit(
            f"stream-setup-aie: no mlir_aie wheel for Python {version[0]}.{version[1]}. "
            f"The pinned build publishes wheels for: {supported}."
        )


def _site_packages() -> Path:
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    return Path(paths[0]) if paths else Path(sysconfig.get_paths()["purelib"])


def _mlir_aie_installed() -> bool:
    try:
        importlib.metadata.version("mlir_aie")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _aie_importable() -> bool:
    try:
        return importlib.util.find_spec("aie") is not None
    except (ImportError, ValueError):
        return False


def _steps(with_mlir_aie: bool) -> list[tuple[str, list[str]]]:
    pip = [sys.executable, "-m", "pip", "install"]
    steps: list[tuple[str, list[str]]] = []
    if with_mlir_aie and _mlir_aie_installed():
        steps.append(("mlir_aie + llvm-aie -- already installed, skipping", []))
    elif with_mlir_aie:
        _check_platform()
        steps.append(
            (
                "mlir_aie + llvm-aie (Xilinx wheels via GitHub release index)",
                [*pip, *_AIE_WHEEL_INDICES, _MLIR_AIE_PIN, _LLVM_AIE_PIN],
            )
        )
    steps.append(("xdsl-aie (git, no-deps)", [*pip, "--no-deps", _XDSL_AIE]))
    steps.append(("snax-mlir (git, no-deps)", [*pip, "--no-deps", _SNAX_MLIR]))
    return steps


def _write_pth(dry_run: bool) -> None:
    # A host with its own mlir_aie wheel already exposes `aie` via the wheel's aie.pth; a second
    # path entry would be redundant and could conflict.
    if _aie_importable():
        print("  aie bindings already importable -- not writing a .pth")
        return
    bindings = _site_packages() / "mlir_aie" / "python"
    pth = _site_packages() / _PTH_FILENAME
    if dry_run:
        print(f"  would write {pth} -> {bindings}")
        return
    pth.write_text(f"{bindings}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stream-setup-aie", description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without executing them.")
    parser.add_argument(
        "--with-mlir-aie",
        action="store_true",
        help="Also install the pinned mlir_aie/llvm-aie wheels (skipped when a mlir_aie is already present).",
    )
    args = parser.parse_args(argv)

    py = f"{sys.version_info[0]}.{sys.version_info[1]}"
    print(f"stream-setup-aie: installing stream-dse's AIE codegen dialects for Python {py}")

    for description, cmd in _steps(args.with_mlir_aie):
        print(f"\n==> {description}")
        if not cmd:
            continue
        if args.dry_run:
            print(f"  {' '.join(cmd)}")
            continue
        subprocess.run(cmd, check=True)

    if args.with_mlir_aie:
        print("\n==> mlir_aie python bindings")
        _write_pth(args.dry_run)

    if args.dry_run:
        print("\nDry run complete -- no changes made.")
    else:
        print("\nDone. AIE code generation (enable_codegen=True) is now available in this environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
