//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LeakyReluOp
//===----------------------------------------------------------------------===//

def Onnx_LeakyReluOp : Onnx_Op<"LeakyRelu", [Pure]> {
  let summary = "ONNX LeakyRelu, schema version 16";

  let description = [{
    LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
    output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "0.01">:$alpha
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
