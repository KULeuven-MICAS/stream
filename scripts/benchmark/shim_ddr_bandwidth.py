"""Measure what the shim DMAs of an AIE2P array move to and from DDR, the figures ``bandwidth.dram`` holds.

Each design connects ``columns`` shims to their mem tiles on ``per_column`` channels. The mem tile side loops on one
buffer, so only the shim side paces the transfer. A channel walks its own 4 MB region in column slabs of ``run``
bytes at an 8 KB pitch: every run starts a new row, so no two consecutive runs are contiguous. ``adjacent`` gives the
channels neighbouring slabs of the same rows instead, as a transfer split over columns does; ``slabs`` of them per row,
all the row holds where not given, so that with one the channels read a column block of each row and skip the rest, as
a GEMM's column block of a row-major operand is read. A ``run`` of 8192 reads or writes the region linearly. The rate
is the bytes of 16 passes over the time of one call, the fastest of 11.

    python scripts/benchmark/shim_ddr_bandwidth.py out.jsonl read:8:2:128 both:4:1:8192 adjacent-read:8:1:128:1 ...
    python scripts/benchmark/shim_ddr_bandwidth.py out.jsonl --derive

Needs mlir-aie (``aiecc``) and XRT's ``pyxrt``; run one at a time on an idle device.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PITCH = 8192
ROWS = 512
REGION = PITCH * ROWS // 4
PASSES = 16
ITERS = 11
CYCLES_PER_NS = 1.8
MAX_D0_WORDS = 512
WHOLE_ARRAY_COLUMNS = 8
CEILING_CHANNELS = 4  # each way, the fewest that reach the ceiling
FULL_RATE_RUN = 1024  # the shortest run that reaches the contiguous rate
SIDES = {"read": (("a", "in", "MM2S", "S2MM"),), "write": (("b", "out", "S2MM", "MM2S"),)}
SIDES["both"] = SIDES["read"] + SIDES["write"]


@dataclass(frozen=True)
class Case:
    """One design: which directions move, on how many columns and channels each, at which run."""

    mode: str
    columns: int
    per_column: int
    run: int
    adjacent: bool = False
    slabs: int = 0

    @property
    def channels(self) -> int:
        return self.columns * self.per_column

    @property
    def moved_bytes(self) -> int:
        share = self.slabs * self.channels * self.run / PITCH if self.slabs else 1
        return int(PASSES * REGION * 4 * self.channels * len(SIDES[self.mode]) * share)

    @classmethod
    def parse(cls, text: str) -> Case:
        mode, columns, per_column, run, *slabs = text.split(":")
        adjacent = mode.startswith("adjacent-")
        if slabs and not adjacent:
            raise ValueError("slabs per row apply to adjacent runs")
        return cls(mode.removeprefix("adjacent-"), int(columns), int(per_column), int(run), adjacent, *map(int, slabs))


def access_pattern(case: Case, channel: int) -> tuple[int, list[int], list[int]]:
    """Offset, sizes and strides in words, outer first, of one pass of one channel over its share of the buffer; the
    outermost size of a walk apart repeats it ``PASSES`` times."""
    words = case.run // 4
    if case.adjacent:
        if case.channels * case.run > PITCH:
            raise ValueError("the channels' slabs do not fit in one row")
        slabs = case.slabs or PITCH // (case.channels * case.run)
        return channel * words, [case.channels, slabs, ROWS, words], [REGION, case.channels * words, PITCH // 4, 1]
    if case.run >= PITCH:
        return (
            channel * REGION,
            [PASSES, ROWS, PITCH // 4 // MAX_D0_WORDS, MAX_D0_WORDS],
            [0, PITCH // 4, MAX_D0_WORDS, 1],
        )
    if words > MAX_D0_WORDS:
        raise ValueError(f"a {case.run} B run exceeds a shim BD's innermost size")
    return channel * REGION, [PASSES, PITCH // case.run, ROWS, words], [0, words, PITCH // 4, 1]


def design(case: Case) -> str:
    """The MLIR of ``case``: shim-to-mem-tile flows, free-running mem tile BDs and the runtime sequence."""
    chans = [(c, k) for c in range(case.columns) for k in range(case.per_column)]
    total = REGION * case.channels
    lines = ["module {", "  aie.device(npu2) {"]
    for c in range(case.columns):
        lines += [f"    %s{c} = aie.tile({c}, 0)", f"    %m{c} = aie.tile({c}, 1)"]
    for c, k in chans:
        for _, name, shim, _ in SIDES[case.mode]:
            ends = (f"%s{c}", f"%m{c}") if shim == "MM2S" else (f"%m{c}", f"%s{c}")
            lines.append(f"    aie.flow({ends[0]}, DMA : {k}, {ends[1]}, DMA : {k})")
            lines.append(
                f'    %{name}{c}_{k}_buf = aie.buffer(%m{c}) {{sym_name = "{name}{c}_{k}_buf"}} : memref<4096xi32>'
            )
            lines.append(f"    aie.shim_dma_allocation @{name}{c}_{k}(%s{c}, {shim}, {k})")
    for c in range(case.columns):
        starts = [(mem, k, name) for k in range(case.per_column) for _, name, _, mem in SIDES[case.mode]]
        lines.append(f"    %md{c} = aie.memtile_dma(%m{c}) {{")
        for i, (mem, k, name) in enumerate(starts):
            after = f"^c{i + 1}" if i + 1 < len(starts) else "^end"
            if i:
                lines.append(f"  ^c{i}:")
            lines += [
                f"      %d{i} = aie.dma_start({mem}, {k}, ^b{i}, {after})",
                f"    ^b{i}:",
                f"      aie.dma_bd(%{name}{c}_{k}_buf : memref<4096xi32> offset = 0 len = 4096)",
                f"      aie.next_bd ^b{i}",
            ]
        lines += ["    ^end:", "      aie.end", "    }"]
    lines.append(f"    aie.runtime_sequence(%a: memref<{total}xi32>, %b: memref<{total}xi32>) {{")
    for _ in range(PASSES if case.adjacent else 1):
        waits = []
        for j, (c, k) in enumerate(chans):
            offset, sizes, strides = access_pattern(case, j)
            for arg, name, _, _ in SIDES[case.mode]:
                lines.append(
                    f"      aiex.npu.dma_memcpy_nd(%{arg}[0, 0, 0, {offset}][{', '.join(map(str, sizes))}]"
                    f"[{', '.join(map(str, strides))}]) {{id = {len(waits) % 8} : i64, metadata = @{name}{c}_{k}, "
                    f"issue_token = true}} : memref<{total}xi32>"
                )
                waits.append(f"{name}{c}_{k}")
        lines += [f"      aiex.npu.dma_wait {{symbol = @{name}}}" for name in waits]
    lines += ["    }", "  }", "}"]
    return "\n".join(lines) + "\n"


def compile_design(case: Case, workdir: Path) -> Path:
    """Build ``case``'s xclbin and instructions in ``workdir``."""
    (workdir / "design.mlir").write_text(design(case))
    flags = ["--get-xclbin", "--get-npu-insts", "--xclbin-name=design.xclbin", "--npu-insts-name=insts.bin"]
    subprocess.run(["aiecc", *flags, "design.mlir"], cwd=workdir, check=True, capture_output=True)
    return workdir


def seconds_per_call(workdir: Path, words: int) -> float:
    """The fastest of ``ITERS`` calls of the design in ``workdir``, after two warm-up calls."""
    import pyxrt as xrt  # noqa: PLC0415 - only a measurement needs XRT, not --derive

    device = xrt.device(0)
    xclbin = xrt.xclbin(str(workdir / "design.xclbin"))
    device.register_xclbin(xclbin)
    context = xrt.hw_context(device, xclbin.get_uuid())
    kernel = xrt.kernel(context, xclbin.get_kernels()[0].get_name())
    insts = np.fromfile(workdir / "insts.bin", dtype=np.uint32)
    instr = xrt.bo(device, insts.nbytes, xrt.bo.flags.cacheable, kernel.group_id(1))
    instr.write(insts.view(np.uint8), 0)
    instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    buffers = [xrt.bo(device, words * 4, xrt.bo.flags.host_only, kernel.group_id(g)) for g in (3, 4)]
    times = []
    for _ in range(ITERS + 2):
        start = time.perf_counter()
        state = kernel(3, instr, len(insts), *buffers).wait()
        times.append(time.perf_counter() - start)
        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"the design did not complete: {state}")
    return min(times[2:])


def measure(case: Case) -> dict:
    """``case`` and the GB/s it moves."""
    with tempfile.TemporaryDirectory() as tmp:
        seconds = seconds_per_call(compile_design(case, Path(tmp)), REGION * case.channels)
    return {**asdict(case), "seconds": seconds, "gb_per_s": case.moved_bytes / seconds / 1e9}


def bits_per_cycle(gb_per_s: float) -> float:
    return round(gb_per_s * 8 / CYCLES_PER_NS, 1)


def derive(records: list[dict]) -> dict:
    """``bandwidth.dram`` from a sweep: per direction, the best of 8 and 16 channels walking runs apart, and for
    adjacent runs the best of one channel per column reading or writing one column block of each row; the ceiling,
    the median of linear reads and writes together on 4 or more channels each way."""
    best, adjacent = (_best(records, kind) for kind in (False, True))
    both = [
        r["gb_per_s"]
        for r in records
        if r["mode"] == "both" and r["run"] >= PITCH and r["columns"] * r["per_column"] >= CEILING_CHANNELS
    ]
    linear = [best[(side, PITCH)] for side in ("read", "write")]
    return {
        "ceiling": bits_per_cycle(statistics.median(both)),
        "contiguous": bits_per_cycle(statistics.mean(linear)),
        "strided": {
            side: {
                run: bits_per_cycle(rate)
                for (mode, run), rate in sorted(best.items())
                if mode == side and run < FULL_RATE_RUN
            }
            for side in ("read", "write")
        },
        "adjacent": {
            side: {run: bits_per_cycle(rate) for (mode, run), rate in sorted(adjacent.items()) if mode == side}
            for side in ("read", "write")
        },
    }


def _best(records: list[dict], adjacent: bool) -> dict[tuple[str, int], float]:
    """Per direction and run, the fastest of the whole array's records walking runs apart, or of its records moving
    one column block of each row on one channel per column."""
    walked = [
        r
        for r in records
        if r["adjacent"] == adjacent
        and r["columns"] == WHOLE_ARRAY_COLUMNS
        and (r.get("slabs") == 1 and r["per_column"] == 1 if adjacent else not r.get("slabs"))
    ]
    return {
        (r["mode"], r["run"]): max(s["gb_per_s"] for s in walked if (s["mode"], s["run"]) == (r["mode"], r["run"]))
        for r in walked
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="JSON lines, appended to")
    parser.add_argument("cases", nargs="*", help="[adjacent-]{read,write,both}:columns:per_column:run")
    parser.add_argument("--derive", action="store_true", help="print bandwidth.dram from the records in out")
    args = parser.parse_args()
    if args.derive:
        print(json.dumps(derive([json.loads(line) for line in args.out.read_text().splitlines()]), indent=1))
        return
    with args.out.open("a") as out:
        for text in args.cases:
            record = measure(Case.parse(text))
            print(json.dumps(record), flush=True)
            out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
