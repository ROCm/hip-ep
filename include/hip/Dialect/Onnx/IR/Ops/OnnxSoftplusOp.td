//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SoftplusOp
//===----------------------------------------------------------------------===//

def Onnx_SoftplusOp : Onnx_Op<"Softplus", [Pure]> {
  let summary = "ONNX Softplus, schema version 22";

  let description = [{
    Softplus takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
    the tensor elementwise.
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
