//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MatMulIntegerOp
//===----------------------------------------------------------------------===//

def Onnx_MatMulIntegerOp : Onnx_Op<"MatMulInteger", [Pure]> {
  let summary = "ONNX MatMulInteger, schema version 10";

  let description = [{
    Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$A,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$B,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$a_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$b_zero_point
  );

  let results = (outs
    TensorOf<[I32]>:$Y
  );
}
