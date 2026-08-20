# Measured against predicted, SwiGLU prefill on Strix

First closed loop between the MILP's latency estimate and hardware trace, for
`swiglu_prefill_stream` k=1 (256/512/2048, npu2, 4x8). The point here is the method and
where the two disagree, not the absolute numbers, which are one design on one part.

## The two sides

Prediction, from `outputs/<experiment>/tetra/slot_latency_breakdown.yaml`:

| | cycles |
|---|---|
| `iter_step` (compute per iteration) | 574 |
| `overlap` | 1040 |
| `latency_per_iteration` | 1614 |
| `total_latency` | 2352144 |

Measurement, from the core trace decoded to Perfetto JSON (64 KiB buffer, one traced
core, 1.25 GHz):

| | |
|---|---|
| kernel invocations paired | 276 |
| kernel duration, median | **575 cycles** |
| kernel duration, min / max | 198 / 575 |
| `LOCK_STALL` total | 626325 cycles |
| `INSTR_VECTOR` total | 50012 cycles |
| trace window | 800138 cycles (640 us) |

Dispatch wall clock, untraced, best of three: **1173.5 us** = 1466875 cycles.

## Where the model is right

**The compute estimate is essentially exact.** Predicted `iter_step` 574 cycles against a
measured median kernel duration of 575. That is 0.2% on the quantity the cost model is
most directly responsible for, and it says the per-core roofline and the kernel's own
cost are being modelled correctly.

## Where it is wrong

**Stalling dominates and is under-modelled.** `LOCK_STALL` accounts for 626325 of the
800138 cycles in the traced window, 78%. Spread over the 276 invocations that is ~2269
cycles of stall per invocation against a predicted `overlap` of 1040, so the waiting is
roughly 2.2x what the model expects.

**End to end the model is conservative by 1.60x**: 2352144 predicted against 1466875
measured. Note this runs the *opposite* way to the per-invocation stall gap, so the two
are not the same error seen twice. Reconciling them needs the iteration count the model
assumes to be checked against the trace, which the 64 KiB window is too short to do.

## Reading these numbers honestly

The trace covers **55%** of the dispatch. A 64 KiB buffer fills before the run ends, so
every total above is a lower bound and only the per-invocation statistics are safe to
quote. A larger buffer would fix this: 1 MiB currently fails to build and that is worth
chasing before any calibration work leans on totals.

Only **one core** is traced. Whole-array k=5 cannot be traced at all -- the design
saturates the stream switches and trace packet flows have nowhere to route, failing with
`Unable to find a legal routing` even at one tile per group. So this is one core's view
of a 32-core design, and the stall figure in particular may not be representative of
cores on different columns.

`INSTR_EVENT_0`/`INSTR_EVENT_1` pairing assumes the two events bracket one kernel call.
276 pairs from 277 starts and 276 ends is consistent with that, but the pairing is by
arrival order, so a dropped packet would silently shift every subsequent span.

## What to do with this

The split matters more than either number: compute is modelled well, waiting is not. That
points calibration effort at the transfer and dependency model rather than the core cost
model, and it is the kind of conclusion a wall-clock comparison alone could never reach --
a single 1.60x number would have suggested scaling the whole estimate, which would have
made the compute term wrong to fix the stall term.

## Reproducing

```
IRON_TRACE_SIZE=65536 IRON_TRACE_NTILES=1 pytest iron/operators/swiglu_prefill_stream -k "k_1 and iter0"
```
then decode with `aie.utils.trace.parse`, passing MLIR that has been through
`--aie-trace-to-config --aie-trace-pack-reg-writes --aie-inline-trace-config` (the parser
reads `aiex.npu.write32`, not the `aie.trace` dialect).
