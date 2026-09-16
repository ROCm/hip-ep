//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GeluOp
//===----------------------------------------------------------------------===//

def Onnx_GeluOp : Onnx_Op<"Gelu", [Pure]> {
  let summary = "ONNX Gelu, schema version 20";

  let description = [{
    Gelu takes one input data (Tensor<T>) and produces one
    output data (Tensor<T>) where the gaussian error linear units function,
    $y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
    If the attribute "approximate" is set to "tanh", the function estimation,
    $y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
    to the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    DefaultValuedStrAttr<StrAttr, "none">:$approximate
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
