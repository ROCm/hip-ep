//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AsinhOp
//===----------------------------------------------------------------------===//

def Onnx_AsinhOp : Onnx_Op<"Asinh", [Pure]> {
  let summary = "ONNX Asinh, schema version 22";

  let description = [{
    Calculates the hyperbolic arcsine of the given input tensor element-wise.
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
