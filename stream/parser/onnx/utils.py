from onnx import TensorProto, ValueInfoProto
from xdsl.dialects.builtin import FixedBitwidthType, MemRefType, bf16, f32, i8, i16, i32
from xdsl.dialects.memref import AllocOp, SubviewOp
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
    memref_type = MemRefType(operand_type, shape)
    memref_source = AllocOp([], [], memref_type)
    # Create subview
    subview = SubviewOp.from_static_parameters(
        source=memref_source,
        source_type=memref_type,
        offsets=[0 for _ in shape],
        sizes=shape,
        strides=[1 for _ in shape],
    )
    return Tensor(
        name=name,
        operand_type=operand_type,
        shape=shape,
        subview=subview,
    )
