//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ReciprocalOp
//===----------------------------------------------------------------------===//

def Onnx_ReciprocalOp : Onnx_Op<"Reciprocal", [Pure]> {
  let summary = "ONNX Reciprocal, schema version 13";

  let description = [{
    Reciprocal takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
    the tensor elementwise.
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
