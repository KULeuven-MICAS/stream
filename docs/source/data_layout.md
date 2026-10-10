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

A memory's efficiency for a contiguous run comes from the `bandwidth` model of its core or port (see
[Hardware](hardware.md)): a table of measured rates per run length, or, where nothing is measured, its `burst`, the
bytes it moves per access, of which a shorter run uses only its share; a direction's measured rates take precedence
over the burst. A memory with neither moves every run at its full rate, and its layouts cost nothing. The run a
transfer writes is measured in the whole tensor, also where its target holds only a tile of it.

## In the results

The performance report of an allocation lists, under `layouts`, every transfer that lays its tile out anew on its chosen
route (`source_order`, `target_order`, outermost first) and the contiguous bytes per run on each side
(`read_run_bytes`, `write_run_bytes`); the typed `AllocationIR` carries them in `performance.layouts`.
