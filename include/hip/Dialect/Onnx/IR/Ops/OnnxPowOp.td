//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_PowOp
//===----------------------------------------------------------------------===//

def Onnx_PowOp : Onnx_Op<"Pow", [Pure]> {
  let summary = "ONNX Pow, schema version 15";

  let description = [{
    Pow takes input data (Tensor<T>) and exponent Tensor, and
    produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
    is applied to the data tensor elementwise.
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>,
      TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>,
      TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Z
  );
}
