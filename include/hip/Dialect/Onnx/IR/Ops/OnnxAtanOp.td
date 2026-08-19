//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AtanOp
//===----------------------------------------------------------------------===//

def Onnx_AtanOp : Onnx_Op<"Atan", [Pure]> {
  let summary = "ONNX Atan, schema version 22";

  let description = [{
    Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
