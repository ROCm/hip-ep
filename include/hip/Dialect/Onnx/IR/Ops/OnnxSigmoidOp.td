//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SigmoidOp
//===----------------------------------------------------------------------===//

def Onnx_SigmoidOp : Onnx_Op<"Sigmoid", [Pure]> {
  let summary = "ONNX Sigmoid, schema version 13";

  let description = [{
    Sigmoid takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
    tensor elementwise.
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
