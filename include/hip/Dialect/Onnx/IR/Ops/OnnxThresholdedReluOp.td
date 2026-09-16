//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ThresholdedReluOp
//===----------------------------------------------------------------------===//

def Onnx_ThresholdedReluOp : Onnx_Op<"ThresholdedRelu", [Pure]> {
  let summary = "ONNX ThresholdedRelu, schema version 22";

  let description = [{
    ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
    is applied to the tensor elementwise.
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
