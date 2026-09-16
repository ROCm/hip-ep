//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_UpsampleOp
//===----------------------------------------------------------------------===//

def Onnx_UpsampleOp : Onnx_Op<"Upsample", [Pure]> {
  let summary = "ONNX Upsample, schema version 9";

  let description = [{
    Upsample the input tensor.
    Each dimension value of the output tensor is:
      output_dimension = floor(input_dimension * scale).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$X,
    TensorOf<[F32]>:$scales,
    DefaultValuedStrAttr<StrAttr, "nearest">:$mode
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$Y
  );
}
