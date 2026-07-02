from onnx import TensorProto, ValueInfoProto
from xdsl.dialects.builtin import FixedBitwidthType, bf16, f32, i8, i16, i32
from zigzag.parser.onnx.utils import OnnxTensorType

from stream.workload.workload import Tensor


def onnx_tensor_to_tensor(tensor: ValueInfoProto | OnnxTensorType | TensorProto, name: str | None = None) -> Tensor:
    name = tensor.name if name is None else name
    onnx_type_to_xdsl_type: dict[int, FixedBitwidthType] = {
        TensorProto.FLOAT: f32,
        TensorProto.BFLOAT16: bf16,
        TensorProto.INT8: i8,
        TensorProto.INT16: i16,
        TensorProto.INT32: i32,
    }
    if isinstance(tensor, ValueInfoProto):
        operand_type = onnx_type_to_xdsl_type[tensor.type.tensor_type.elem_type]
        shape = tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim)
    elif isinstance(tensor, OnnxTensorType):
        operand_type = onnx_type_to_xdsl_type[tensor.elem_type]
        shape = tuple(tensor.shape)
    else:
        operand_type = onnx_type_to_xdsl_type[tensor.data_type]
        shape = tuple(tensor.dims)
    return Tensor.create(name=name, operand_type=operand_type, shape=shape)
