# Workload

The recommended way of defining an algorithmic workload is through an
onnx model. An onnx model can contain multiple operator types, which in
the context of ML are often referred to as layers, some of which are
automatically recognised and parsed by Stream. Alternatively, the layers
can be manually defined for more customization.

Stream handles workloads in a similar way than the [original ZigZag
project](https://kuleuven-micas.github.io/zigzag/workload.html).
Additionally, Stream supports more onxx operators, but it also requires
slightly different layer attributes in the manual layer definition.

## ONNX models

### Supported operators

A complete list of all onnx operators can be found
[here](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

Following operators are supported by Stream and will automatically be
parsed into `LayerNode` objects when using your onnx model within the
framework:

-   [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)
-   [QLinearConv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv)
-   [MatMul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul)
-   [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm)
-   [MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool)
-   [AveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)
-   [GlobalMaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool)
-   [GlobalAveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool)
-   [Reshape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape)
-   [Flatten](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten)
-   [Add](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add)
-   [Mul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul)
-   [Transpose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose)
-   [LpNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization)

All other operators will be parsed into a `DummyNode` object, which is
assumed to not be accelerateable, incurring 0 hardware cost. If you have
an onnx operator you would like to see supported, feel free to [open an
issue](https://github.com/KULeuven-MICAS/stream/issues/new) or manually
add it yourself in the
[ONNXModelParserStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/parser/onnx/model.py)
taking into account the [Contributing Guidelines](contribute.md).

### Saving weights as external data

If your onnx model is rather large, and you want to avoid having it
inside of your Stream repo, you can save it with external data, which
saves the weights as an external file, which can be discarded as Stream
doesn't require the weight values. You can do so as follows:

``` python
import onnx
model = onnx.load('my_model_with_internal_data.onnx')
onnx.save_model(
    model,
    'my_model_with_external_data.onnx',
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location='external_data.data',
    size_threshold=1024,
    convert_attribute=False
)
# Optional: Remove the 'external_data_filename' file if you don't need the data
# When loading model, add extra flag as follows:
model = onnx.load_model('my_model_with_internal_data.onnx', load_external_data=False)
```

### Shape inference

Stream requires an inferred onnx model, as it needs to know the shapes
of all intermediate tensors to correctly infer the layer shapes. If you
have an onnx model that is not shape inferred, you can do so by the
following commands:

``` python
import onnx
from onnx import shape_inference
model = onnx.load("my_model.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "my_inferred_model.onnx")
```

## Manual layer definition

It is also possible to manually define your own workload layers. In that
case there the `main.py` file should be executed instead of
`main_onnx.py`. Moreover, the workload file should be provided as input
together with the accelerator, thus there is no onnx model and mapping
file loaded. The mapping information is inserted for each layer
alongside the layer shape definition, identically to how it was defined
in the mapping file.

Each layer definition is represented as a dict which should have the
following attributes:

-   **operator_type**: Specification of the layer operator type (e.g.
    `Conv`, `MaxPool`).
-   **equation**: The operational equation for this layer. The
    dimensions should be small letters, where as the operands are large
    letters. 'O' should always be used for the output operand, the
    input operands can be named freely.
-   **dimension_relations**: The relationship between different
    dimensions present in the equation. This is often used in
    convolutional layers, where there is a relationship between the
    spatial input indices and the spatial output indices through the
    stride and with the filter indices through the dilation rate.
-   **loop_dim_size**: The size of the different dimensions present in
    the equation. Dimensions defined (i.e. on the left hand side) in the
    dimension_relations are not to be provided and are inferred
    automatically.
-   **operand_precision**: The bit precision of the different operands
    present in the equation. 'O' should always be used, which
    represents the partial output precision. 'O_final' represents the
    final output precision.
-   **operand_source**: The layer id the input operands of this layer
    come from. This is important to correctly build the NN graph edges.
-   **memory_operand_links**: The link between the virtual memory
    operands and the actual algorithmic operands. For more information,
    read the hardware readme.
-   **padding**: Padding for each dimension (i.e. IX and IY). The first
    value is the left or upper padding, the second value is the right or
    lower padding for IX resp. IY

The following loop notation has to be used to describe a layer of the
workload (see loop notation in [this
paper](https://ieeexplore.ieee.org/document/9360462)):

-   **B**: Batch size
-   **K**: Output channels
-   **C**: Input channels
-   **OY**: Output rows
-   **OX**: Output columns
-   **FY**: Kernel rows
-   **FX**: Kernel columns

An **example of this manual layer defintion** can be found at:
[inputs/examples/workloads/resnet18.yaml](https://github.com/KULeuven-MICAS/stream/blob/master/stream/inputs/examples/workload/resnet18.yaml).
