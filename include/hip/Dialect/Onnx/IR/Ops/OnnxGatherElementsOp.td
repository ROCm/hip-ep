//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GatherElementsOp
//===----------------------------------------------------------------------===//

def Onnx_GatherElementsOp : Onnx_Op<"GatherElements", [Pure]> {
  let summary = "ONNX GatherElements, schema version 13";

  let description = [{
    GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
    and an optional attribute `axis` that identifies an axis of `data`
    (by default, the outer-most axis, that is axis 0). It is an indexing operation
    that produces its output by indexing into the input data tensor at index
    positions determined by elements of the `indices` tensor.
    Its output shape is the same as the shape of `indices` and consists of one value
    (gathered from the `data`) for each element in `indices`.

    For instance, in the 3-D case (r = 3), the output produced is determined
    by the following equations:
    ```
    out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
    out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
    out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
    ```

    This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

    Example 1:
    ```
    data = [
        [1, 2],
        [3, 4],
    ]
    indices = [
        [0, 0],
        [1, 0],
    ]
    axis = 1
    output = [
        [1, 1],
        [4, 3],
    ]
    ```
    Example 2:
    ```
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    indices = [
        [1, 2, 0],
        [2, 0, 0],
    ]
    axis = 0
    output = [
        [4, 8, 3],
        [7, 2, 3],
    ]
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
