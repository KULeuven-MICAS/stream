"""Small non-AIE SwiGLU workload builder for CI testing.

Graph: 4 nodes (last_gemm_down=False)
  Gemm_Left  : [seq_len, embedding_dim] @ [embedding_dim, hidden_dim] -> [seq_len, hidden_dim]
  Gemm_Right : [seq_len, embedding_dim] @ [embedding_dim, hidden_dim] -> [seq_len, hidden_dim]
  Silu       : [seq_len, hidden_dim] -> [seq_len, hidden_dim]  (com.example custom function-op)
  Elt_Mul    : [seq_len, hidden_dim] x [seq_len, hidden_dim] -> [seq_len, hidden_dim]

CI dimensions (confirmed run-green on all 8 non-AIE hardware boards):
  seq_len=1        (M dimension of each Gemm; analogous to batch=1 in make_2_conv)
  embedding_dim=16 (K dimension: input feature width)
  hidden_dim=32    (N dimension: intermediate feature width)

Expected ComputationNode count: 4 (Gemm_Left, Gemm_Right, Silu, Elt_Mul) -- confirmed by running.

Adapted from the AIE swiglu generator (make_onnx_swiglu.py), last_gemm_down=False branch.
The logic is copied in -- this module imports nothing from the AIE input tree (ROADMAP criterion 3).
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

SEQ_LEN = 1
EMBEDDING_DIM = 16
HIDDEN_DIM = 32


def _clear_tensor_data(tensor: TensorProto) -> None:
    """Remove raw weight data from a tensor, keeping only shape and type metadata."""
    for field in ("float_data", "double_data", "int32_data", "int64_data", "uint64_data", "raw_data"):
        tensor.ClearField(field)


def make_small_swiglu_workload(output_dir=None) -> str:
    """Build a small 4-node SwiGLU ONNX (last_gemm_down=False) and return its path.

    The graph is Gemm_Left, Gemm_Right, a custom com.example Silu function-op, and an
    element-wise Mul (Elt_Mul). No down-projection Gemm. bf16 throughout.

    Args:
        output_dir: Directory to write the ONNX into. Defaults to this module's directory
            (a permanent location -- the path must outlive any tmpdir the CO run uses).

    Returns:
        str: Path to the saved ONNX model.
    """
    # Initializers (weights) -- shapes [embedding_dim, hidden_dim]; data cleared (metadata only).
    w_left = helper.make_tensor(
        "weights_1",
        TensorProto.BFLOAT16,
        [EMBEDDING_DIM, HIDDEN_DIM],
        np.zeros((EMBEDDING_DIM, HIDDEN_DIM)),
    )
    _clear_tensor_data(w_left)
    w_right = helper.make_tensor(
        "weights_2",
        TensorProto.BFLOAT16,
        [EMBEDDING_DIM, HIDDEN_DIM],
        np.zeros((EMBEDDING_DIM, HIDDEN_DIM)),
    )
    _clear_tensor_data(w_right)

    # I/O
    inp = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [SEQ_LEN, EMBEDDING_DIM])
    out = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [SEQ_LEN, HIDDEN_DIM])

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

    # Final element-wise multiply -> output (last_gemm_down=False).
    out_mul = helper.make_node(
        "Mul",
        inputs=["left_swished", "right"],
        outputs=["output"],
        name="Elt_Mul",
    )

    graph = helper.make_graph(
        nodes=[gemm_left, gemm_right, silu_left, out_mul],
        name="SwiGLU",
        inputs=[inp],
        outputs=[out],
        initializer=[w_left, w_right],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
        producer_name="stream-dse",
        producer_version="1.0",
        doc_string="Small SwiGLU (4 nodes, last_gemm_down=False) for CO pipeline CI testing.",
    )

    # Attach the function definition to the model.
    model.functions.extend([silu_func])

    # Shape inference
    inferred = shape_inference.infer_shapes(model)

    # Save to a permanent location (NOT a tmpdir -- the CO run reads this path later).
    dest_dir = output_dir if output_dir is not None else os.path.dirname(__file__)
    onnx_path = os.path.join(dest_dir, f"swiglu_{SEQ_LEN}_{EMBEDDING_DIM}_{HIDDEN_DIM}_no_gemm_down.onnx")
    onnx.save(inferred, onnx_path)
    print(f"SwiGLU ONNX model created: {onnx_path}")

    return onnx_path
