//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_HardSwishOp
//===----------------------------------------------------------------------===//

def Onnx_HardSwishOp : Onnx_Op<"HardSwish", [Pure]> {
  let summary = "ONNX HardSwish, schema version 22";

  let description = [{
    HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
    the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
    where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
