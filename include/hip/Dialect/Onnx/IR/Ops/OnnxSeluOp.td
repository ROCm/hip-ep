//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SeluOp
//===----------------------------------------------------------------------===//

def Onnx_SeluOp : Onnx_Op<"Selu", [Pure]> {
  let summary = "ONNX Selu, schema version 22";

  let description = [{
    Selu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the scaled exponential linear unit function,
    `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
    is applied to the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "1.67326">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0507">:$gamma
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
