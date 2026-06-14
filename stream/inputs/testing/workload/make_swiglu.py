"""Non-AIE SwiGLU workload builder for CI testing.

Graph: 5 nodes
  Gemm_Left  : [seq_len, embedding_dim] @ [embedding_dim, hidden_dim] -> [seq_len, hidden_dim]
  Gemm_Right : [seq_len, embedding_dim] @ [embedding_dim, hidden_dim] -> [seq_len, hidden_dim]
  Silu       : [seq_len, hidden_dim] -> [seq_len, hidden_dim]  (com.example custom function-op)
  Elt_Mul    : [seq_len, hidden_dim] x [seq_len, hidden_dim] -> [seq_len, hidden_dim]
  Gemm_Down  : [seq_len, hidden_dim] @ [hidden_dim, embedding_dim] -> [seq_len, embedding_dim]

The dimensions are parametrized. The default is the tiny committed CI fixture (seq_len=1,
embedding_dim=16, hidden_dim=32), kept for the `just co-swiglu` / `just gen-workloads` convenience
runs. The hardware-matrix swiglu test (tests/test_hardware_combinations.py) instead passes the AIE
`just swiglu` dimensions (seq_len=256, embedding_dim=2048, hidden_dim=8192) and drives the generic
pipeline with a fused intra-core tiling so the whole block is processed layer-fused (one steady-state
tile per solve), not as the full 256x2048x8192 layer.

Weight initializers carry shape metadata only -- their values are cleared (only tensor
sizes matter for cost estimation), so the committed ONNX stays small.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

DEFAULT_SEQ_LEN = 1
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_HIDDEN_DIM = 32


def _clear_tensor_data(tensor: TensorProto) -> None:
    """Remove raw weight data from a tensor, keeping only shape and type metadata."""
    for field in ("float_data", "double_data", "int32_data", "int64_data", "uint64_data", "raw_data"):
        tensor.ClearField(field)


def make_small_swiglu_workload(
    seq_len: int = DEFAULT_SEQ_LEN,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    output_dir=None,
) -> str:
    """Build a 5-node SwiGLU ONNX and return its path.

    The graph is Gemm_Left, Gemm_Right, a custom com.example Silu function-op, an
    element-wise Mul (Elt_Mul), and a down-projection Gemm (Gemm_Down). bf16 throughout.

    Args:
        seq_len: Sequence length (M dimension of each Gemm). Defaults to the tiny CI fixture value.
        embedding_dim: Input/output feature width. Defaults to the tiny CI fixture value.
        hidden_dim: Intermediate feature width. Defaults to the tiny CI fixture value.
        output_dir: Directory to write the ONNX into. Defaults to this module's directory.

    Returns:
        str: Path to the saved ONNX model.
    """
    # Initializers (weights) -- shape metadata only; values cleared.
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
    w_down = helper.make_tensor(
        "weights_3",
        TensorProto.BFLOAT16,
        [hidden_dim, embedding_dim],
        np.zeros((hidden_dim, embedding_dim)),
    )
    _clear_tensor_data(w_down)

    # I/O
    inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [seq_len, embedding_dim])
    out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [seq_len, embedding_dim])

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
    )

    # Define custom SiLU as a Function in a private domain (body = X * Sigmoid(X)).
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

    # Use the function-op as a single node.
    silu_left = helper.make_node(
        "Silu",
        inputs=["left"],
        outputs=["left_swished"],
        name="Silu",
        domain="com.example",
    )

    # Element-wise multiply -> intermediate.
    out_mul = helper.make_node(
        "Mul",
        inputs=["left_swished", "right"],
        outputs=["intermediate"],
        name="Elt_Mul",
    )

    # Down-projection GEMM -> output.
    gemm_down = helper.make_node(
        "Gemm",
        inputs=["intermediate", "weights_3"],
        outputs=["output"],
        name="Gemm_Down",
        transA=0,
        transB=0,
        alpha=1.0,
        beta=1.0,
    )

    graph = helper.make_graph(
        nodes=[gemm_left, gemm_right, silu_left, out_mul, gemm_down],
        name="SwiGLU",
        inputs=[inp],
        outputs=[out],
        initializer=[w_left, w_right, w_down],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="SwiGLU (5 nodes) for CO pipeline CI testing.",
    )

    # Attach the function definition to the model.
    model.functions.extend([silu_func])

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save
    dest_dir = output_dir if output_dir is not None else os.path.dirname(__file__)
    onnx_path = os.path.join(dest_dir, f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}.onnx")
    onnx.save(inferred, onnx_path)
    print(f"SwiGLU ONNX model created: {onnx_path}")

    return onnx_path
