//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ConvIntegerOp
//===----------------------------------------------------------------------===//

def Onnx_ConvIntegerOp : Onnx_Op<"ConvInteger", [Pure]> {
  let summary = "ONNX ConvInteger, schema version 10";

  let description = [{
    The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
    and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$x_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$w_zero_point,
    DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<SI64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    TensorOf<[I32]>:$y
  );
}
