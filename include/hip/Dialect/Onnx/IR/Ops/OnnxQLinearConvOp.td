//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_QLinearConvOp
//===----------------------------------------------------------------------===//

def Onnx_QLinearConvOp : Onnx_Op<"QLinearConv", [Pure]> {
  let summary = "ONNX QLinearConv, schema version 10";

  let description = [{
    The convolution operator consumes a quantized input tensor, its scale and zero point,
    a quantized filter, its scale and zero point, and output's scale and zero point,
    and computes the quantized output. Each scale and zero-point pair must have same shape.
    It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
    Each input or output and its related zero point must have same type.
    When bias is present it must be quantized using scale = input scale * weight scale and
    zero point as 0.
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
    TensorOf<[F32]>:$x_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
    TensorOf<[F32]>:$w_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w_zero_point,
    TensorOf<[F32]>:$y_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y_zero_point,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$B,
    DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<SI64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y
  );
}
