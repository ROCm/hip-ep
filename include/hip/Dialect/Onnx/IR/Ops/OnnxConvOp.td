//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ConvOp
//===----------------------------------------------------------------------===//

def Onnx_ConvOp : Onnx_Op<"Conv", [Pure]> {
  let summary = "ONNX Conv, schema version 22";

  let description = [{
    The convolution operator consumes an input tensor and a filter, and
    computes the output.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$W,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$B,
    DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<SI64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
