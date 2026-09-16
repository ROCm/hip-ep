//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GatherOp
//===----------------------------------------------------------------------===//

def Onnx_GatherOp : Onnx_Op<"Gather", [Pure]> {
  let summary = "ONNX Gather, schema version 13";

  let description = [{
    Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
    entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
    them in an output tensor of rank q + (r - 1).

    It is an indexing operation that indexes into the input `data` along a single (specified) axis.
    Each entry in `indices` produces a `r-1` dimensional slice of the input tensor.
    The entire operation produces, conceptually, a `q`-dimensional tensor of `r-1` dimensional slices,
    which is arranged into a `q + (r-1)`-dimensional tensor, with the `q` dimensions taking the
    place of the original `axis` that is being indexed into.

    The following few examples illustrate how `Gather` works for specific shapes of `data`,
    `indices`, and given value of `axis`:
    | data shape | indices shape | axis | output shape | output equation |
    | --- | --- | --- | --- | --- |
    | (P, Q) | ( )  (a scalar)   | 0 | (Q)       | output[q] = data[indices, q] |
    | (P, Q, R) | ( )  (a scalar)   | 1 | (P, R)       | output[p, r] = data[p, indices, r] |
    | (P, Q) | (R, S) | 0 | (R, S, Q) | output[r, s, q] = data[ [indices[r, s], q] |
    | (P, Q) | (R, S) | 1 | (P, R, S) | output[p, r, s] = data[ p, indices[r, s]] |

    More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1} ]`
    then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2} ] = input[k , j_{0}, ..., j_{r-2} ]`:

    ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    indices = [
        [0, 1],
        [1, 2],
    ]
    output = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
        ],
    ]
    ```

    If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1} ]`
    then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2} ] = input[j_{0}, k, j_{1}, ..., j_{r-2} ]`:

    ```
    data = [
        [1.0, 1.2, 1.9],
        [2.3, 3.4, 3.9],
        [4.5, 5.7, 5.9],
    ]
    indices = [
        [0, 2],
    ]
    axis = 1,
    output = [
            [[1.0, 1.9]],
            [[2.3, 3.9]],
            [[4.5, 5.9]],
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
