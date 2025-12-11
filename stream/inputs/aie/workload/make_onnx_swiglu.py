import os

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype):
    """
    Build an ONNX model for SwiGLU using a custom function-op SiLU in a private domain:
        left  = Gemm(input, w_left)                 # [seq_len,embedding_dim] @ [embedding_dim,hidden_dim] -> [seq_len,hidden_dim]
        right = Gemm(input, w_right)                # [seq_len,embedding_dim] @ [embedding_dim,hidden_dim] -> [seq_len,hidden_dim]
        left_swished = com.example::SiLU(left)      # single node; body = left * Sigmoid(left)
        intermediate = left_swished * right         # elementwise Mul -> [seq_len,hidden_dim]
        output = Gemm(intermediate, w_down)         # [seq_len,hidden_dim] @ [hidden_dim,embedding_dim] -> [seq_len,embedding_dim]

    Args:
        seq_len (int): Sequence length dimension.
        embedding_dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension for intermediate layers.
        in_dtype (str): Input data type.
        out_dtype (str): Output data type.
    Returns:
        onnx.ModelProto: Shape-inferred model.
    """
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
        TensorProto.FLOAT,
        [embedding_dim, hidden_dim],
        np.zeros((embedding_dim, hidden_dim)),
    )
    w_left.ClearField("float_data")
    w_right = helper.make_tensor(
        "weights_2",
        TensorProto.FLOAT,
        [embedding_dim, hidden_dim],
        np.zeros((embedding_dim, hidden_dim)),
    )
    w_right.ClearField("float_data")
    w_down = helper.make_tensor(
        "weights_3",
        TensorProto.FLOAT,
        [hidden_dim, embedding_dim],
        np.zeros((hidden_dim, embedding_dim)),
    )
    w_down.ClearField("float_data")

    # I/O
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [seq_len, embedding_dim])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [seq_len, embedding_dim])

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

    # Final elementwise multiply
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

    graph = helper.make_graph(
        nodes=[gemm_left, gemm_right, silu_left, out_mul, gemm_down],
        name="SwiGLU",
        inputs=[inp],
        outputs=[out],
        initializer=[w_left, w_right, w_down],
    )

    # Build model and register both standard and custom domains
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
        producer_name="swiglu-generator",
        producer_version="1.0",
        doc_string="SwigLU: left Gemm -> SiLU (custom function-op), right Gemm, then elementwise Mul and down projection Gemm.",
    )

    # Attach the function definition to the model
    model.functions.extend([silu_func])

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    onnx_path = os.path.join(os.path.dirname(__file__), f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}.onnx")
    onnx.save(inferred, onnx_path)
    print(f"SWIGLU ONNX model created: {onnx_path}")

    return onnx_path
