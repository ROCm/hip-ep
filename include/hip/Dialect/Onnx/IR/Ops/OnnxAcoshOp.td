//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AcoshOp
//===----------------------------------------------------------------------===//

def Onnx_AcoshOp : Onnx_Op<"Acosh", [Pure]> {
  let summary = "ONNX Acosh, schema version 22";

  let description = [{
    Calculates the hyperbolic arccosine of the given input tensor element-wise.
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
