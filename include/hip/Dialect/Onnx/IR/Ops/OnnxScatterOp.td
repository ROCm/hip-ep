//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ScatterOp
//===----------------------------------------------------------------------===//

def Onnx_ScatterOp : Onnx_Op<"Scatter", [Pure]> {
  let summary = "ONNX Scatter, schema version 9";

  let description = [{
    Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
    into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
    For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
    for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
    data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
    where i and j are loop counters from 0 up to the respective size in `updates` - 1.
    Example 1:
      data = [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
      ]
      indices = [
          [1, 0, 2],
          [0, 2, 1],
      ]
      updates = [
          [1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2],
      ]
      output = [
          [2.0, 1.1, 0.0]
          [1.0, 0.0, 2.2]
          [0.0, 2.1, 1.2]
      ]
    Example 2:
      data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
      indices = [[1, 3]]
      updates = [[1.1, 2.1]]
      axis = 1
      output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$updates,
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
