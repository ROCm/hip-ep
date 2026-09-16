//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LogOp
//===----------------------------------------------------------------------===//

def Onnx_LogOp : Onnx_Op<"Log", [Pure]> {
  let summary = "ONNX Log, schema version 13";

  let description = [{
    Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$input
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output
  );
}
