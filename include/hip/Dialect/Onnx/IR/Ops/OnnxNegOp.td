//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_NegOp
//===----------------------------------------------------------------------===//

def Onnx_NegOp : Onnx_Op<"Neg", [Pure]> {
  let summary = "ONNX Neg, schema version 13";

  let description = [{
    Neg takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where each element flipped sign, y = -x, is applied to
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
