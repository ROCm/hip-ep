//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_EyeLikeOp
//===----------------------------------------------------------------------===//

def Onnx_EyeLikeOp : Onnx_Op<"EyeLike", [Pure]> {
  let summary = "ONNX EyeLike, schema version 22";

  let description = [{
    Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
    tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
    same as the input tensor. The data type can be specified by the 'dtype' argument. If
    'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
    is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
    The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
    TensorProto message and be valid as an output type.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>
    ]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    DefaultValuedAttr<SI64Attr, "0">:$k
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>
    ]>:$output
  );
}
