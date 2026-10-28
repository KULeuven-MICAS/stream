# Data layout

Every copy of a tensor Stream places in a memory has a layout: the order of its axes from outermost to innermost, the
innermost contiguous. Layouts decide how long the contiguous runs are that a transfer reads and writes, and a memory's
bandwidth depends on those runs. Stream chooses them the way an accelerator's compiler and DMAs do, before the
allocation is solved, and the allocation then prices every transfer at the bandwidth its runs get.

## What a kernel needs

A compute core reads an operand's tile through memory words, each feeding the units its array unrolls along. The axes
the core reads together in one word must be innermost in the copy it reads: its *contiguous axes*. Each core's cost
backend says what they are (`OperandLayoutSource.contiguous_axes`). The ZigZag backend reads them off the spatial
mapping it costed: per operand, the axes unrolled widest over the array, such as a convolution's channels or a matmul's
contraction along the rows of a systolic array, and both axes of the block a stationary operand loads. Nothing is
declared by hand; a backend for other hardware implements the same method, and one that does not leaves its copies
as they come.

## How copies are laid out

| Copy | Layout |
|------|--------|
| A model parameter (an ONNX initializer, a weight) | Packed ahead of time for the kernel reading it |
| Any other workload input, and every workload output | Row-major, as the host holds it |
| A node's output | Its contiguous axes innermost, as the core produces it |
| A copy a node reads | Its source with the node's contiguous axes moved innermost, as little reordered as that takes: a transpose the kernel reads as it is moves nothing. Where its readers differ, the first decides, and every copy a multicast makes has one layout |
| A copy in the memory it is copied from | Its source's layout: nothing moves, so nothing can be reordered |
| A copy on its way to another | Its source's layout or the layout of the copy it feeds, whichever lets the two transfers move it faster, and the latter before a hop that stays in one memory |

The layout operators of a model (`Transpose`, `Reshape`, …) are folded into their readers when the workload is parsed
(see [Workload](workload.md#layout-operators)); what they leave is which axis is which, and the layouts above decide the
order the data is in.

## Converting on the fly

A transfer between two layouts converts the tile as it moves it. The data streams in one layout's order: the DMA
gathers in the target's order, reading runs only as long as the innermost axes the two layouts share, or scatters in
the source's order, writing such runs. Stream takes whichever its slower side moves faster, so a transpose out of a
memory that charges for short runs is written scattered into one that does not. An axis of one element places its data
nowhere, so moving it is no conversion.

## What the hardware declares

A memory's efficiency for a contiguous run comes from the `bandwidth` model of the memory or port (see
[Hardware](hardware.md#measured-bandwidth)): a table of measured rates per run length, or, where nothing is measured, its `burst`, the
bytes it moves per access, of which a shorter run uses only its share; a direction's measured rates take precedence
over the burst. A memory with neither moves every run at its full rate, and its layouts cost nothing. Each side's runs
are measured in the buffer it holds: the whole tensor for a workload input or output off chip, the tile itself on chip,
so a tile written into a buffer of its own size is one run in any layout both sides share.

## In the results

The performance report of an allocation lists, under `layouts`, every transfer on its chosen route: the tile it moves
(`tensor`, `tile_shape`, `full_shape`, `element_bits`), the memories it reads and writes (`source_memories`, `target_memories`),
the axis orders of both copies (`source_order`, `target_order`, outermost first), whether it lays its tile out anew
(`converts`), and the contiguous bytes per run on each side (`read_run_bytes`, `write_run_bytes`); the typed
`AllocationIR` carries them in `performance.layouts`.

Each transfer also says which layout the data streams in (`streams_in`: the target's when the DMA gathers, else the
source's), names the tile's axes in the dims of a node that touches it (`axes`, as a mapping writes them, `D3 + D5` for
a window), lists the steady-state loops the copy goes through (`iterations`, outermost first: each one's dim, trip
count, kind, effect on the copy, reuse and halo, and the tile axis it advances by `step` elements per trip, None for a
loop the copy does not index), and describes both sides (`read`, `write`):

| Field | Meaning |
|-------|---------|
| `extent` | The buffer the copy sits in, per axis: the whole tensor off chip, the tile on chip |
| `strides` | Elements between neighbours along each axis of that buffer, from its axis order |
| `access` | The `[size, stride]` loops, outermost first, in elements, the DMA walks the tile in on that side, in the order it streams; loops of one element dropped and a loop continuing the one inside it merged, as a DMA descriptor writes them |
| `memory`, `port` | The memory and port the side passes (`measured` for a memory with a measured bandwidth) |
| `bits_per_cycle`, `burst_bytes` | That port's contiguous rate and the bytes it moves per access, where declared |
| `banks` | The banks of the memory, where its level declares them (see [Hardware](hardware.md)) |

A 64x64 weight tile of a 512x64 row-major tensor transposed into a core reads `[[64, 1], [64, 64]]`, one element
at a stride of 64 per step, and writes `[[4096, 1]]`, one contiguous run.
