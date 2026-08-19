//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RangeOp
//===----------------------------------------------------------------------===//

def Onnx_RangeOp : Onnx_Op<"Range", [Pure]> {
  let summary = "ONNX Range, schema version 11";

  let description = [{
    Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
    up to `limit` (exclusive).

    The number of elements in the output of range is computed as below:

    ```
    number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
    ```

    The pseudocode determining the contents of the output is shown below:

    ```
    for(int i=0; i<number_of_elements; ++i) {
      output[i] =  start + (i * delta);
    }
    ```

    Example 1

    ```
    Inputs: start = 3, limit = 9, delta = 3
    Output: [3, 6]
    ```

    Example 2

    ```
    Inputs: start = 10, limit = 4, delta = -2
    Output: [10, 8, 6]
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>
    ]>:$start,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>
    ]>:$limit,
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>
    ]>:$delta
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>,
      TensorOf<[I64]>
    ]>:$output
  );
}
