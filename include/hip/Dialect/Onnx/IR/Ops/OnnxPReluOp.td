//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_PReluOp
//===----------------------------------------------------------------------===//

def Onnx_PReluOp : Onnx_Op<"PRelu", [Pure]> {
  let summary = "ONNX PRelu, schema version 16";

  let description = [{
    PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
    output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
    `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
    This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>
    ]>:$slope
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>
    ]>:$Y
  );
}
