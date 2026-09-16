//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DequantizeLinearOp
//===----------------------------------------------------------------------===//

def Onnx_DequantizeLinearOp : Onnx_Op<"DequantizeLinear", [Pure]> {
  let summary = "ONNX DequantizeLinear, schema version 25";

  let description = [{
    The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
    full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
    must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
    a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
    See QuantizeLinear for details on quantization granularity.

    `x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
    `int32`, there's no zero point (zero point is supposed to be 0).
    `zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
    for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
    is the same as `x_scale`. The output type also determines the precision of the multiplication operation.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>,
      TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>:$x,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F8E8M0FNU]>
    ]>:$x_scale,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>,
      TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>, NoneType
    ]>:$x_zero_point,
    DefaultValuedAttr<SI64Attr, "1">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$block_size,
    DefaultValuedAttr<SI64Attr, "0">:$output_dtype
  );

  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y
  );
}
