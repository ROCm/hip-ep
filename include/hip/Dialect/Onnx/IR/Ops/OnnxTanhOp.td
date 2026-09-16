//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_TanhOp
//===----------------------------------------------------------------------===//

def Onnx_TanhOp : Onnx_Op<"Tanh", [Pure]> {
  let summary = "ONNX Tanh, schema version 13";

  let description = [{
    Calculates the hyperbolic tangent of the given input tensor element-wise.
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
