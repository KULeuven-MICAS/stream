import os

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def _clear_tensor_data(tensor: TensorProto) -> None:
    """Remove raw weight data from a tensor, keeping only shape and type metadata."""
    for field in ("float_data", "double_data", "int32_data", "int64_data", "uint64_data", "raw_data"):
        tensor.ClearField(field)


def make_swiglu_workload(
    seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype, last_gemm_down: bool = True, output_dir=None
):
    """SwiGLU as ONNX, its SiLU a function op in a private domain: SiLU(Gemm(input, w_left)) * Gemm(input, w_right),
    then a Gemm by w_down with ``last_gemm_down``. Written into ``output_dir``, by default this module's directory;
    returns the model."""
    dest_dir = output_dir if output_dir is not None else os.path.dirname(__file__)
    # Validate and convert shapes
    seq_len = int(seq_len)
    embedding_dim = int(embedding_dim)
    hidden_dim = int(hidden_dim)

    # Get data type sizes
    dtype_size_map = {"f32": 32, "f16": 16, "bf16": 16, "i8": 8}
    if in_dtype not in dtype_size_map or out_dtype not in dtype_size_map:
        raise ValueError("Unsupported data type. Supported types: float32, float16, bfloat16, int8.")
    act_size = dtype_size_map[in_dtype]
    weight_size = dtype_size_map[in_dtype]
    output_size = dtype_size_map[out_dtype]

    # Initializers (weights)
    w_left = helper.make_tensor(
        "weights_1",
        TensorProto.BFLOAT16,
        [embedding_dim, hidden_dim],
        np.zeros((embedding_dim, hidden_dim)),
    )
    _clear_tensor_data(w_left)
    w_right = helper.make_tensor(
        "weights_2",
        TensorProto.BFLOAT16,
        [embedding_dim, hidden_dim],
        np.zeros((embedding_dim, hidden_dim)),
    )
    _clear_tensor_data(w_right)

    # I/O
    inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [seq_len, embedding_dim])

    # Branch GEMMs
    gemm_left = helper.make_node(
        "Gemm",
        inputs=["input", "weights_1"],
        outputs=["left"],
        name="Gemm_Left",
        transA=0,
        transB=0,
        alpha=1.0,
        beta=1.0,
        act_size=act_size,
        weight_size=weight_size,
        output_size=output_size,
    )
    gemm_right = helper.make_node(
        "Gemm",
        inputs=["input", "weights_2"],
        outputs=["right"],
        name="Gemm_Right",
        transA=0,
        transB=0,
        alpha=1.0,
        beta=1.0,
        act_size=act_size,
        weight_size=weight_size,
        output_size=output_size,
    )

    # Define custom SiLU as a Function in private domain
    silu_func = helper.make_function(
        domain="com.example",
        fname="Silu",
        inputs=["X"],
        outputs=["Y"],
        nodes=[
            helper.make_node("Sigmoid", inputs=["X"], outputs=["S"], name="Sigmoid_in_fn"),
            helper.make_node("Mul", inputs=["X", "S"], outputs=["Y"], name="Mul_in_fn"),
        ],
        opset_imports=[helper.make_opsetid("", 17)],
        doc_string="SiLU(X) = X * Sigmoid(X)",
    )

    # Use the function-op as a single node
    silu_left = helper.make_node(
        "Silu",
        inputs=["left"],
        outputs=["left_swished"],
        name="Silu",
        domain="com.example",
        act_size=act_size,
        weight_size=0,
        output_size=output_size,
    )

    if not last_gemm_down:
        # Final elementwise multiply
        out_mul = helper.make_node(
            "Mul",
            inputs=["left_swished", "right"],
            outputs=["output"],
            name="Elt_Mul",
            act_size=act_size,
            weight_size=0,
            output_size=output_size,
        )
        # If last gemm down is not needed, output directly from the elementwise mul
        out_mul.output[0] = "output"
        out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [seq_len, hidden_dim])
        graph = helper.make_graph(
            nodes=[gemm_left, gemm_right, silu_left, out_mul],
            name="SwiGLU",
            inputs=[inp],
            outputs=[out],
            initializer=[w_left, w_right],
        )
        doc_string = "SwigLU: left Gemm -> SiLU, right Gemm, then elementwise Mul."
        onnx_path = os.path.join(dest_dir, f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}_no_gemm_down.onnx")
    else:
        w_down = helper.make_tensor(
            "weights_3",
            TensorProto.BFLOAT16,
            [hidden_dim, embedding_dim],
            np.zeros((hidden_dim, embedding_dim)),
        )
        _clear_tensor_data(w_down)
        # Intermediate elementwise multiply
        out_mul = helper.make_node(
            "Mul",
            inputs=["left_swished", "right"],
            outputs=["intermediate"],
            name="Elt_Mul",
            act_size=act_size,
            weight_size=0,
            output_size=output_size,
        )
        # Final down projection gemm
        gemm_down = helper.make_node(
            "Gemm",
            inputs=["intermediate", "weights_3"],
            outputs=["output"],
            name="Gemm_Down",
            transA=0,
            transB=0,
            alpha=1.0,
            beta=1.0,
            act_size=act_size,
            weight_size=weight_size,
            output_size=output_size,
        )
        out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [seq_len, embedding_dim])

        graph = helper.make_graph(
            nodes=[gemm_left, gemm_right, silu_left, out_mul, gemm_down],
            name="SwiGLU",
            inputs=[inp],
            outputs=[out],
            initializer=[w_left, w_right, w_down],
        )
        doc_string = "SwigLU: left Gemm -> SiLU, right Gemm, then elementwise Mul and down proj Gemm."
        onnx_path = os.path.join(dest_dir, f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}.onnx")

    # Build model and register both standard and custom domains
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
        producer_name="swiglu-generator",
        producer_version="1.0",
        doc_string=doc_string,
    )

    # Attach the function definition to the model
    model.functions.extend([silu_func])

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    onnx.save(inferred, onnx_path)
    print(f"SWIGLU ONNX model created: {onnx_path}")

    return onnx_path
