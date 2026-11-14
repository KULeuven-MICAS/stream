"""tpu_like_quad_core with its matrix cores declaring a partial-sum format, for the tests of partial sums across
cores."""

import shutil
from pathlib import Path

TPU_QUAD = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def tpu_quad(tmp_path: Path, partial_sum: str | None) -> str:
    """tpu_like_quad_core in ``tmp_path``, its matrix cores declaring ``partial_sum`` (None: none) next to their int32
    accumulator."""
    shutil.copytree(Path(TPU_QUAD).parent / "cores", tmp_path / "cores")
    if partial_sum:
        core = tmp_path / "cores" / "tpu_like.yaml"
        core.write_text(
            core.read_text().replace("accumulator: int32", f"accumulator: int32\n  partial_sum: {partial_sum}")
        )
    (tmp_path / "chip.yaml").write_text(Path(TPU_QUAD).read_text())
    return str(tmp_path / "chip.yaml")
