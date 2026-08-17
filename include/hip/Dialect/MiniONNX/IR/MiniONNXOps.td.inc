//********************************************************
//   MiniONNX Operation Definitions
//   Auto-generated from ONNX schemas
//
//   To regenerate: python3 gen_mini_onnx.py --ops <ops>
//********************************************************

def MiniONNX_MatMulOp : MiniONNX_Op<"MatMul",
  [Pure]> {
  let summary = "ONNX MatMul operation";
  let description = [{
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$Y
  );
}

def MiniONNX_AddOp : MiniONNX_Op<"Add",
  [Pure]> {
  let summary = "ONNX Add operation";
  let description = [{
  Performs element-wise binary addition (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
}

def MiniONNX_ConstantOp : MiniONNX_Op<"Constant",
  [Pure]> {
  let summary = "ONNX Constant operation";
  let description = [{
  This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
  or value_* must be specified.
  }];
  let arguments = (ins
    OptionalAttr<AnyAttr>:$sparse_value,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<F32Attr>:$value_float,
    OptionalAttr<F32ArrayAttr>:$value_floats,
    OptionalAttr<SI64Attr>:$value_int,
    OptionalAttr<I64ArrayAttr>:$value_ints,
    OptionalAttr<StrAttr>:$value_string,
    OptionalAttr<StrArrayAttr>:$value_strings
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
}

def MiniONNX_ExpandOp : MiniONNX_Op<"Expand",
  [Pure]> {
  let summary = "ONNX Expand operation";
  let description = [{
  Broadcast the input tensor following the given shape and the broadcast rule.
  The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
  Dimensions are right alignment;
  Two corresponding dimensions must have the same value, or one of them is equal to 1.
  Also, this operator is similar to numpy.broadcast_to(input, shape),
  but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
  It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
  or the shape.ndim < input.shape.ndim.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I64]>:$shape
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
}

