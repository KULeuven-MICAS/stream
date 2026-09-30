# Kernel library

A kernel library tells Stream what a target's compiled kernels accept, what object each one links against, and what one call costs. It belongs to whoever builds the kernels. An accelerator file names its library with `kernel_library: <path>`, relative to the accelerator file, and `SolveOptions(kernel_library=...)` replaces it with a TOML or YAML path, a mapping, or a `KernelLibrary`. `stream/inputs/aie/kernels/aie2p.toml` is the library the example AIE accelerators name.

```toml
[family.matmul]
ops_per_cycle = 151.0
mac = { m = 8, k = 8, n = 8 }

[family.vector]
ops_per_cycle = 16.0

[kernel.matmul_bf16_bf16]
family = "matmul"
object = "mm_{m}_{k}_{n}.o"
dims = [{ name = "k", divisor = 8 }, { name = "n", divisor = 16 }, { name = "m", divisor = 16 }]
cycles = [{ m = 64, k = 64, n = 64, cycles = 1595.0 }]

[kernel.silu_bf16_size]
family = "vector"
object = "silu.o"
dims = [{ name = "n", runtime = true, keep_whole = true }, { name = "m", runtime = true }]
per_op = { cycles = 2503.0, ops = 2048 }
```

## Families

`matmul` and `vector` are the two families. `ops_per_cycle` prices a kernel of the family that has no measured call. The matmul family's `mac` is the MAC tile the matmul kernels are compiled for, which also sets the tiling an elementwise operand keeps beside a matmul.

## Kernels

A kernel is keyed by its symbol, and a GEMM by its symbol without the shape suffix.

| Key | Meaning |
|---|---|
| `family` | `matmul` or `vector`. |
| `object` | The object the kernel links against; `{name}` fields take the call's size along that dimension. |
| `dims` | The call's dimensions, innermost first. |
| `cycles` | Measured calls, each giving every dimension's size and the cycles one call took. An unmeasured shape is priced from the measured call nearest in size. |
| `per_op` | Cycles and operations of one reference call, for a kernel with no shape-keyed measurement. |
| `source` | The kernel source, for the library's own build. |

## Dimensions

| Key | Meaning |
|---|---|
| `name` | The kernel field holding the size. |
| `runtime` | The size is a call argument, so a larger intra-core tile still reaches the kernel. |
| `fixed` | The only size the source is compiled at. |
| `blocks` | The sizes the source is compiled and measured at. |
| `divisor` | Every legal size is a multiple of this. |
| `keep_whole` | The dimension stays a tiling level even when one call covers all of it. |
