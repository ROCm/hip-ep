//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MatMulOp
//===----------------------------------------------------------------------===//

def Onnx_MatMulOp : Onnx_Op<"MatMul", [Pure]> {
  let summary = "ONNX MatMul, schema version 13";

  let description = [{
    Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$A,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$B
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
      TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
