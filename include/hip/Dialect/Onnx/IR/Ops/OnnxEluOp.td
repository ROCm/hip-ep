//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_EluOp
//===----------------------------------------------------------------------===//

def Onnx_EluOp : Onnx_Op<"Elu", [Pure]> {
  let summary = "ONNX Elu, schema version 22";

  let description = [{
    Elu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
    0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
