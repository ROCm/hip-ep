//********************************************************
//   MiniONNX Operation Definitions
//   Auto-generated from ONNX schemas
//
//   To regenerate: python3 gen_mini_onnx.py --ops all
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
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ArrayRef<NamedAttribute>":$attributes)>
  ];
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
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ArrayRef<NamedAttribute>":$attributes)>
  ];
}

