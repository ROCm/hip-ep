//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SqrtOp
//===----------------------------------------------------------------------===//

def Onnx_SqrtOp : Onnx_Op<"Sqrt", [Pure]> {
  let summary = "ONNX Sqrt, schema version 13";

  let description = [{
    Square root takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the square root is, y = x^0.5, is applied to
    the tensor elementwise. If x is negative, then it will return NaN.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
