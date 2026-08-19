//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ReluOp
//===----------------------------------------------------------------------===//

def Onnx_ReluOp : Onnx_Op<"Relu", [Pure]> {
  let summary = "ONNX Relu, schema version 14";

  let description = [{
    Relu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
    the tensor elementwise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>,
      TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>,
      TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
