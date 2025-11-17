import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

def make_swiglu_workload(input_shape, out_channels):
    """
    Build an ONNX model for SwigLU using a custom function-op SiLU in a private domain:
        left  = Gemm(input, W1)                 # [x,y] @ [y,n] -> [x,n]
        right = Gemm(input, W2)                 # [x,y] @ [y,n] -> [x,n]
        left_silu = com.example::SiLU(left)     # single node; body = left * Sigmoid(left)
        result = left_silu * right              # elementwise Mul -> [x,n]

    Args:
        input_shape (tuple): (x, y) for a 2D input.
        out_channels (int): n, output channels for both Gemms.
    Returns:
        onnx.ModelProto: Shape-inferred model.
    """
    # Validate and unpack shapes
    if not (isinstance(input_shape, (tuple, list)) and len(input_shape) == 2):
        raise ValueError("input_shape must be a tuple/list of length 2 like (x, y).")
    x, y = int(input_shape[0]), int(input_shape[1])
    n = int(out_channels)

    # Initializers (weights)
    # Omit the weight values by creating dummy weight initializers, just for shape definition
    B1 = helper.make_tensor(
        "B1",
        TensorProto.FLOAT,
        [y, n],
        np.zeros((y, n)),
    )
    B1.ClearField("float_data")
    B2 = helper.make_tensor(
        "B1",
        TensorProto.FLOAT,
        [y, n],
        np.zeros((y, n)),
    )
    B2.ClearField("float_data")

    # I/O
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [x, y])
    out = helper.make_tensor_value_info("result", TensorProto.FLOAT, [x, n])

    # Branch GEMMs
    gemm_left = helper.make_node(
        "Gemm",
        inputs=["input", "B1"], outputs=["left"],
        name="Gemm_Left",
        transA=0, transB=0, alpha=1.0, beta=1.0
    )
    gemm_right = helper.make_node(
        "Gemm",
        inputs=["input", "B2"], outputs=["right"],
        name="Gemm_Right",
        transA=0, transB=0, alpha=1.0, beta=1.0
    )

    # Define custom SiLU as a Function in private domain (appears as a single node in the graph)
    # Body uses standard ONNX ops so built-in shape inference can see through it.
    silu_func = helper.make_function(
        domain="com.example",
        fname="SiLU",
        inputs=["X"],
        outputs=["Y"],
        nodes=[
            helper.make_node("Sigmoid", inputs=["X"], outputs=["S"], name="Sigmoid_in_fn"),
            helper.make_node("Mul", inputs=["X", "S"], outputs=["Y"], name="Mul_in_fn"),
        ],
        opset_imports=[helper.make_opsetid("", 17)],
        doc_string="SiLU(X) = X * Sigmoid(X)"
    )

    # Use the function-op as a single node
    silu_left = helper.make_node(
        "SiLU",
        inputs=["left"], outputs=["left_silu"],
        name="SiLU_Left",
        domain="com.example"
    )

    # Final elementwise multiply
    out_mul = helper.make_node(
        "Mul",
        inputs=["left_silu", "right"], outputs=["result"],
        name="SwigLU_Out"
    )

    graph = helper.make_graph(
        nodes=[gemm_left, gemm_right, silu_left, out_mul],
        name="SwigLU",
        inputs=[inp],
        outputs=[out],
        initializer=[B1, B2],
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
        doc_string="SwigLU: left Gemm -> SiLU (custom function-op), right Gemm, then elementwise Mul."
    )

    # Attach the function definition to the model
    model.functions.extend([silu_func])

    # Shape inference (propagates through the function body)
    inferred = shape_inference.infer_shapes(model)

    # Save
    onnx_path = f"swiglu_{x}_{y}_{n}.onnx"
    onnx.save(inferred, onnx_path)
    return onnx_path

# Example usage:
m = make_swiglu_workload((32, 128), 256)