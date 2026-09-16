//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SwishOp
//===----------------------------------------------------------------------===//

def Onnx_SwishOp : Onnx_Op<"Swish", [Pure]> {
  let summary = "ONNX Swish, schema version 24";

  let description = [{
    Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
    where $Swish(x) = x * sigmoid(alpha * x)$.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>
    ]>:$Y
  );
}
