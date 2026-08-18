//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ResizeOp
//===----------------------------------------------------------------------===//

def Onnx_ResizeOp : Onnx_Op<"Resize", [Pure]> {
  let summary = "ONNX Resize, schema version 19";

  let description = [{
    Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
    Each dimension value of the output tensor is:
    ```
    output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
    ```
    if input \"sizes\" is not specified.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType
    ]>:$roi,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$scales,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
    DefaultValuedAttr<SI64Attr, "0">:$antialias,
    OptionalAttr<I64ArrayAttr>:$axes,
    DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
    DefaultValuedAttr<F32Attr, "-0.75">:$cubic_coeff_a,
    DefaultValuedAttr<SI64Attr, "0">:$exclude_outside,
    DefaultValuedAttr<F32Attr, "0.0">:$extrapolation_value,
    DefaultValuedStrAttr<StrAttr, "stretch">:$keep_aspect_ratio_policy,
    DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
    DefaultValuedStrAttr<StrAttr, "round_prefer_floor">:$nearest_mode
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$Y
  );
}
