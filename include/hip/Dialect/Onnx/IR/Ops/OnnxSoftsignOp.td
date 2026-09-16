//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SoftsignOp
//===----------------------------------------------------------------------===//

def Onnx_SoftsignOp : Onnx_Op<"Softsign", [Pure]> {
  let summary = "ONNX Softsign, schema version 22";

  let description = [{
    Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
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
