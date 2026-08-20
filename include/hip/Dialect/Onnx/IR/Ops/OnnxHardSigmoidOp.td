//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_HardSigmoidOp
//===----------------------------------------------------------------------===//

def Onnx_HardSigmoidOp : Onnx_Op<"HardSigmoid", [Pure]> {
  let summary = "ONNX HardSigmoid, schema version 22";

  let description = [{
    HardSigmoid takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
    is applied to the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "0.2">:$alpha,
    DefaultValuedAttr<F32Attr, "0.5">:$beta
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
