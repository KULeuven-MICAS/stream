# Kernel library

A kernel library tells Stream what a target's compiled kernels accept, who provides each one, and what one call costs. It belongs to whoever builds the kernels. An accelerator file names its library with `kernel_library: <path>`, relative to the accelerator file, and `SolveOptions(kernel_library=...)` replaces it with a TOML or YAML path, a mapping, or a `KernelLibrary`. `stream/inputs/aie/kernels/aie2p.toml` is the library the example AIE accelerators name.

```toml
[family.matmul]
ops_per_cycle = 151.0
mac = { m = 8, k = 8, n = 8 }

[family.vector]
ops_per_cycle = 16.0

[kernel.matmul_bf16_bf16]
family = "matmul"
binding = "stream.compiler.kernels.mlir_aie:mm"
dims = [{ name = "k", divisor = 8 }, { name = "n", divisor = 16 }, { name = "m", divisor = 16 }]
cycles = [{ m = 64, k = 64, n = 64, cycles = 1595.0 }]

[kernel.silu_bf16_size]
family = "vector"
binding = "stream.compiler.kernels.mlir_aie:silu"
dims = [{ name = "n", runtime = true, keep_whole = true }, { name = "m", runtime = true }]
per_op = { cycles = 2503.0, ops = 2048 }
```

## Families

`matmul` and `vector` are the two families. `ops_per_cycle` prices a kernel of the family that has no measured call. The matmul family's `mac` is the MAC tile the matmul kernels are compiled for, which also sets the tiling an elementwise operand keeps beside a matmul.

## Kernels

A kernel is keyed by the name Stream calls it by.

| Key | Meaning |
|---|---|
| `family` | `matmul` or `vector`. |
| `binding` | The provider, `"module:function"`, called with the target `npu` and the call's dimensions; it returns the kernel's binding. |
| `object` | For a kernel no provider binds: the object it links against, its symbol and signature being Stream's own. |
| `dims` | The call's dimensions, innermost first. |
| `cycles` | Measured calls, each giving every dimension's size and the cycles one call took. An unmeasured shape is priced from the measured call nearest in size. |
| `per_op` | Cycles and operations of one reference call, for a kernel with no shape-keyed measurement. |

## Bindings

A binding is what a kernel's provider declares for one call: its symbol (`name`), the object that defines it (`object_file_name`), and its arguments (`arg_types()`), numpy array types `np.ndarray[(n,), np.dtype[t]]` or numpy scalar types. mlir-aie's `ExternalFunction` is one, and `stream/compiler/kernels/mlir_aie.py` binds the AIE kernels to the `aie.iron.kernels` factories. A call Stream makes beside the kernel's own, such as the `zero` that clears a GEMM's output, is an attribute of the binding.

Codegen calls each binding's symbol, checks the call against its arguments, and declares it with `link_with` naming its object. It writes the bindings a design used to `kernels.json` beside the design, and `stream.compiler.kernels.binding.load_bindings` turns that file back into one binding per object, for a host to build the objects from.

## Dimensions

| Key | Meaning |
|---|---|
| `name` | The kernel field holding the size. |
| `runtime` | The size is a call argument, so a larger intra-core tile still reaches the kernel. |
| `fixed` | The only size the source is compiled at. |
| `blocks` | The sizes the source is compiled and measured at. |
| `divisor` | Every legal size is a multiple of this. |
| `keep_whole` | The dimension stays a tiling level even when one call covers all of it. |
