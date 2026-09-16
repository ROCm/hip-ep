//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AsinOp
//===----------------------------------------------------------------------===//

def Onnx_AsinOp : Onnx_Op<"Asin", [Pure]> {
  let summary = "ONNX Asin, schema version 22";

  let description = [{
    Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
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
