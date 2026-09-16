//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MishOp
//===----------------------------------------------------------------------===//

def Onnx_MishOp : Onnx_Op<"Mish", [Pure]> {
  let summary = "ONNX Mish, schema version 22";

  let description = [{
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    Perform the linear unit element-wise on the input tensor X using formula:

    ```
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    ```
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
