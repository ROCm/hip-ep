//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LpNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_LpNormalizationOp : Onnx_Op<"LpNormalization", [Pure]> {
  let summary = "ONNX LpNormalization, schema version 22";

  let description = [{
    Given a matrix, apply Lp-normalization along the provided axis.
    The output is computed as: `output = input / Lp_norm(input, axis)`.
    When the Lp norm is zero (i.e., all elements along the axis are zero),
    the output is defined to be zero to avoid division by zero.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    DefaultValuedAttr<SI64Attr, "-1">:$axis,
    DefaultValuedAttr<SI64Attr, "2">:$p
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
