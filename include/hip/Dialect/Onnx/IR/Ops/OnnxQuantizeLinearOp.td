//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_QuantizeLinearOp
//===----------------------------------------------------------------------===//

def Onnx_QuantizeLinearOp : Onnx_Op<"QuantizeLinear", [Pure]> {
  let summary = "ONNX QuantizeLinear, schema version 25";

  let description = [{
    The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
    low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
    granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

    Saturation is done according to:
    - uint16: [0, 65535]
    - int16: [-32768, 32767]
    - uint8: [0, 255]
    - int8: [-128, 127]
    - uint4: [0, 15]
    - int4: [-8, 7]
    - uint2: [0, 3]
    - int2: [-2, 1]

    For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

    `y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
    formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
    `x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
    `y_scale`, unless the `precision` attribute is specified.

    There are three supported quantization granularities, determined by the shape of `y_scale`.
    In all cases, `y_zero_point` must have the same shape as `y_scale`.
    - Per-tensor (per-layer) quantization: `y_scale` is a scalar.
    - Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
     `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
    - Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
      blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
      `(D0, ..., ceil(Di/B), ..., Dn)`.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>
    ]>:$x,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>,
      TensorOf<[F8E8M0FNU]>
    ]>:$y_scale,
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[UI<2>]>, TensorOf<[I<2>]>, NoneType
    ]>:$y_zero_point,
    DefaultValuedAttr<SI64Attr, "1">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$block_size,
    DefaultValuedAttr<SI64Attr, "0">:$output_dtype,
    DefaultValuedAttr<SI64Attr, "0">:$precision,
    DefaultValuedAttr<SI64Attr, "1">:$saturate
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$y
  );
}
