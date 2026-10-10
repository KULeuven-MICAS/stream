# Workload

A workload is the neural-network computation you want to map. Stream ingests workloads through **pluggable frontends**; **ONNX** is the default and most-validated one. A frontend walks the model graph and turns recognised operators into the internal, affine computation nodes that the rest of the pipeline tiles, costs, schedules and allocates.

The ONNX frontend lives in `stream/parser/onnx/` (`stream/parser/onnx/model.py` is the dispatch table). What follows reflects exactly what that code parses today.

---

## ONNX in, affine computation graph out

Stream loads an ONNX model, runs **shape inference** on it, and converts each node:

- A **supported operator** becomes a `ComputationNode` — it has a real cost and is placed on a core.
- A **layout-only operator** (`Reshape`, `Transpose`, …) is folded into the nodes reading its output (see [Layout operators](#layout-operators)); only one that cannot be folded becomes a `FusionEdge`, a boundary between fusion groups.
- An **unrecognised operator** raises `NotImplementedError`. Stream does *not* silently drop unknown ops; you either register a parser for it or remove it from the model.

### Supported operators

The dispatch table (`ONNXModelParser.OP_TYPE_TO_PARSER`) recognises:

| ONNX op | Becomes | Notes |
|---------|---------|-------|
| `Conv` | ComputationNode | Convolution: strides, dilations and padding (or `auto_pad`) per axis, `group`, and the bias as a third input. |
| `Gemm` | ComputationNode | General matrix multiply (also matrix-vector). |
| `MatMul` | ComputationNode | Batched matrix multiply. |
| `MaxPool` | ComputationNode | Max pooling, through the same per-axis window as `Conv`. |
| `GlobalAveragePool` | ComputationNode | Global average pooling. |
| `BatchNormalization` | ComputationNode | Batch normalisation. |
| `Softmax`, `LayerNormalization`, `LpNormalization` | NormalizationNode | Reduce-then-broadcast; the reduced axis is a fusion barrier, the other axes stay parallel. |
| `Slice`, `Gather` | ComputationNode | Data movement / indexing (e.g. a KV cache), carrying the moved region. |
| `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Relu`, `Silu`, `Gelu`, `Sigmoid`, `Tanh` | ComputationNode | Element-wise (unary and binary, NumPy broadcast). |
| `Cast` | ComputationNode | Element-wise conversion to another element type. |
| `QuantizeLinear`, `DequantizeLinear` | folded, or ComputationNode | See [Element types and quantized models](#element-types-and-quantized-models). |
| `Einsum` | ComputationNode | One loop per index letter. Two operands contract; one operand summing letters away is a `ReduceSum`; one only reordering its axes is a transpose. |
| `ReduceSum`, `ReduceMean`, `ReduceMax` | ComputationNode | Reduces the axes its output drops (`axes` as input or attribute, `keepdims`). A `ReduceSum` of a matmul's or einsum's output folds into it as a contraction. |
| `Flatten`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze` | folded, or FusionEdge | See [Layout operators](#layout-operators). |

To support a new operator, register a parser (see [Extending ingestion](#extending-ingestion)).

### The affine representation

Every `ComputationNode` carries an **`operand_mapping`**: one affine map (`AffineMap`) per operand, from the node's iteration space to that operand's indices. A MatMul `ik,kj->ij`, for example, maps its three operands with `(i,k)`, `(k,j)` and `(i,j)`. Everything the pipeline needs is *derived* from these maps rather than hard-coded per op:

- **Loop dimensions and sizes** come from the maps and the operand shapes. A dim no operand axis indexes alone, such
  as a pool's kernel, takes its extent from the node's `window_extents`, `(dim, extent)` pairs its parser sets.
- **Reduction dimensions** are the iteration dimensions that index an input but not the output (a contraction, like `k` above); the rest are parallel.
- **Operand access relations** classify how each operand is read — a plain affine access, a piecewise-affine access (masked / windowed regions), or a data-dependent access (gather / routing) — which is what fusion analysis reasons over.

Because the representation is uniform, adding an operator is a matter of giving its affine maps; the tiling, cost, fusion and dedup passes consume it unchanged.

### Shape inference is required

Stream needs the shape of every intermediate tensor to derive each node's loop dimensions. The frontend calls `onnx.shape_inference.infer_shapes` for you, but the model must carry enough type/shape information for inference to succeed. If you build a model by hand, infer shapes before saving:

```python
import onnx
from onnx import shape_inference

model = onnx.load("my_model.onnx")
onnx.save(shape_inference.infer_shapes(model), "my_model_inferred.onnx")
```

### Layout operators

A transpose or reshape moves no data an accelerator has to compute: a compiler lays the tensor out so its reader can
index the original (a reshape is another view of the same buffer, a transpose an operand layout or a DMA access
pattern). Stream does the same with the access maps: the node reading a layout operator's output reads its input
instead, through the composed map.

- A **transpose** permutes the reader's indices.
- A **reshape that splits an axis** (`[S, D]` as `[S, H, D/H]`) indexes it with the row-major combination of the new
  axes, and the axis is then refined into those axes in every node and tensor that has it, so each node indexes each
  axis with one loop.
- A **reshape that merges axes** (`[H, S, D/H]` to `[S, D]`) splits the reader's loop over the merged axis into one
  loop per original axis: an output projection reading merged heads contracts heads and head dimension.

The three ways frameworks write multi-head attention (PyTorch's reshapes and transposes, JAX's einsums, a per-head
projection summed over the heads) therefore parse to the same nodes. A layout operator that regroups elements across
axes (`[6, 4]` viewed as `[4, 6]`), or whose merged axis its reader walks other than with one loop, is materialized as
a `FusionEdge`: the tensor crosses memory between two fusion groups. How the folded tensors are laid out in memory, and what
converting between layouts costs, is decided afterwards (see [Data layout](data_layout.md)).

### Element types and quantized models

Every tensor keeps the element type the model gives it (`float32`, `float16`, `bfloat16`, `int8`, `uint8`, `int16`,
`int32`), and nodes read and write at those widths. A model exported at deployment precision is therefore costed at
that precision; an fp32 model is costed in fp32, so cast it first if the hardware computes in narrower types.
The example workloads are at the precision these models are deployed in: the CNNs (ResNet-18, FSRCNN) in int8,
ResNet-18 with int32 biases, and the LLM blocks (SwiGLU, attention) in bf16.

A quantized model in QDQ form marks its int8 tensors with `QuantizeLinear` and `DequantizeLinear` pairs. These
describe element types rather than computations, and Stream folds them the way a deployment compiler does:

- a `DequantizeLinear`'s readers read its quantized input directly, so a convolution or matmul runs on int8;
- a `QuantizeLinear` folds into the node producing its input when nothing else reads that input, so the producer
  writes the quantized tensor itself (the requantization at the end of its computation).

A conversion with nothing to fold into, such as quantizing the model's fp32 input or dequantizing its output, stays in
the graph as a `Cast` node and is costed like any element-wise op, on a core whose `operator_types` include `Cast`.
Partial sums are kept at the accumulator precision of the core computing them (see
[Hardware](hardware.md#operand-precision)).

### Weights are not needed — clear them

Stream only uses tensor **shapes and dtypes** for cost modelling; it never reads weight *values*. Keep your committed ONNX small by clearing the initializer data. Note that the data may live in any of several fields depending on dtype (bf16 weights, for instance, pack into `int32_data`, **not** `float_data`), so clear them all:

```python
for field in ("float_data", "double_data", "int32_data",
              "int64_data", "uint64_data", "raw_data"):
    tensor.ClearField(field)
```

This is exactly what the bundled workload builders do — the committed example ONNX are only a few hundred bytes because their weights are cleared.

For very large models you can alternatively keep weights in an external file (`onnx.save_model(..., save_as_external_data=True)`) and load with `load_external_data=False`; Stream works fine without the external data present.

---

## Building a workload programmatically

The repo ships small workloads as ready-to-use ONNX fixtures under `stream/inputs/testing/workload/`, generated by Python builders in the same directory. `just gen-workloads` regenerates them.

**`make_2_conv.py`** — two chained `Conv` layers. The committed fixture `2conv_1_8_32_32_16_32_3.onnx` is `[1,8,32,32] → Conv(16) → Conv(32) → [1,32,32,32]`.

**`make_swiglu.py`** — a 5-node SwiGLU block: two parallel `Gemm`s, a `Silu` activation, an element-wise `Mul`, and a down-projection `Gemm`. The committed fixture is `swiglu_1_16_32.onnx`.

A minimal builder looks like this:

```python
import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 8, 32, 32])
out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 16, 32, 32])

w = helper.make_tensor("weights", TensorProto.BFLOAT16, [16, 8, 3, 3],
                       np.zeros((16, 8, 3, 3)))
for field in ("float_data", "double_data", "int32_data",
              "int64_data", "uint64_data", "raw_data"):
    w.ClearField(field)          # keep shape + dtype, drop values

conv = helper.make_node("Conv", ["input", "weights"], ["output"],
                        name="Conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1])

graph = helper.make_graph([conv], "OneConv", [inp], [out], initializer=[w])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(shape_inference.infer_shapes(model), "one_conv.onnx")
```

Once saved, run it through the pipeline like any other workload:

```python
from stream.api import evaluate_mapping

evaluate_mapping("stream/inputs/examples/hardware/tpu_like_quad_core.yaml", "one_conv.onnx", "outputs/one-conv")
```

Stream also ships parameterized reference blocks for the building blocks of modern models — attention, GQA, a Mamba-style recurrence, SwiGLU/MLP, RMSNorm, MoE — as affine workload graphs you can build directly (`stream.workload.blocks.build_block`) for experiments that do not start from an ONNX file.

---

## Extending ingestion

Stream is extended through registries, so you can add coverage from your own package without editing (or forking) the tree. Each has an entry-point group of the same name, so an installed package is discovered automatically.

- **A new operator** — add a parser (a subclass of `OnnxOperatorParser`) and register it with `stream.parser.onnx.model.register_onnx_parser("MyOp", MyParser)`, or declare it under the `stream.onnx_parsers` entry-point group. A registered parser overrides the built-in table, so higher-level ops (for example a fused attention op) can lower into several affine nodes at once.
- **A new ingestion format** — implement the `WorkloadFrontend` protocol (`stream.frontends`) and register it under `stream.frontends`. `stream.frontends.load_workload` then picks the first frontend that accepts the source. ONNX and an optional `torch.export` frontend ship in-tree.
- **Reference blocks** — register block builders under `stream.workload_blocks`.
- **Operator decompositions** — register a `node → Workload` decomposer under `stream.decompositions` to expose an operator's affine sub-operators (the granularity fusion analysis and a matmul-array + vector-unit cost view want).

See [Getting Started](getting-started.md) for the full run flow and [Mapping](mapping.md) for how the operators in your workload get matched to cores.
