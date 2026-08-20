//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DetOp
//===----------------------------------------------------------------------===//

def Onnx_DetOp : Onnx_Op<"Det", [Pure]> {
  let summary = "ONNX Det, schema version 22";

  let description = [{
    Det calculates determinant of a square matrix or batches of square matrices.
    Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
    and the inner-most 2 dimensions form square matrices.
    The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
    e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
