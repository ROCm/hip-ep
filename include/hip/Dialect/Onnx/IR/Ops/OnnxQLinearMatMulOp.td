//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_QLinearMatMulOp
//===----------------------------------------------------------------------===//

def Onnx_QLinearMatMulOp : Onnx_Op<"QLinearMatMul", [Pure]> {
  let summary = "ONNX QLinearMatMul, schema version 21";

  let description = [{
    Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
    and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
    For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
    Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
    (per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
    or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
    an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
    for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
    have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
    Production must never overflow, and accumulation may overflow if and only if in 32 bits.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$a,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$a_scale,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$a_zero_point,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$b,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$b_scale,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$b_zero_point,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y_scale,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$y_zero_point
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>,
      TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>
    ]>:$y
  );
}
