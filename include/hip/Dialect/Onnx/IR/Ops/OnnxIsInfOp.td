//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_IsInfOp
//===----------------------------------------------------------------------===//

def Onnx_IsInfOp : Onnx_Op<"IsInf", [Pure]> {
  let summary = "ONNX IsInf, schema version 20";

  let description = [{
    Map infinity to true and other values to false.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>
    ]>:$X,
    DefaultValuedAttr<SI64Attr, "1">:$detect_negative,
    DefaultValuedAttr<SI64Attr, "1">:$detect_positive
  );

  let results = (outs
    TensorOf<[I1]>:$Y
  );
}
