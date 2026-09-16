//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_IsNaNOp
//===----------------------------------------------------------------------===//

def Onnx_IsNaNOp : Onnx_Op<"IsNaN", [Pure]> {
  let summary = "ONNX IsNaN, schema version 20";

  let description = [{
    Returns which elements of the input are NaN.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>
    ]>:$X
  );

  let results = (outs
    TensorOf<[I1]>:$Y
  );
}
