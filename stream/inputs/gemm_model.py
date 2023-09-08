import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx.backend.test.case.node import expect,_extract_value_info
from onnx import helper, shape_inference
from typing import Any, Sequence
import numpy as np
import onnx
import onnxruntime
class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
    
        self.inputSize = 288
        self.outputSize = 1
        self.hiddenSize = 1
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) 
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        
    def forward(self, X):
        out= torch.matmul(X, self.W1) 

        return out
        


def expect_gagan(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)
  


node = onnx.helper.make_node(
    "MatMul",
    inputs=["a", "b"],
    outputs=["c"],
)
model=Neural_Network()

dummy_input=torch.randn(288,288)
torch.onnx.export(model, dummy_input, "test_gemm.onnx")

