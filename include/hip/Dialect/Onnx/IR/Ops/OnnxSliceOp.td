//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SliceOp
//===----------------------------------------------------------------------===//

def Onnx_SliceOp : Onnx_Op<"Slice", [Pure]> {
  let summary = "ONNX Slice, schema version 13";

  let description = [{
    Produces a slice of the input tensor along multiple axes. Similar to numpy:
    https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

    Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
    of its input `data` tensor.

    An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
    in `[0, ... r-1]` where `r = rank(input)` as follows:

    If `axes` are omitted, they are set to `[0, ..., r-1]`.
    If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

    The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
    `dims` are the dimensions of `input` and `steps[i] = 1`.

    All negative elements of `axes` are made non-negative by adding `r` to them, where
    `r =rank(input)`.

    All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
    where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
    `starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
    and `[0, dims[axes[i]]-1]` for negative stepping.

    The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
    accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
    `ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
    is clamped to `[-1, dims[axes[i]]-1]`.

    Finally, `steps[axes[i]] = steps[i]`.

    For slicing to the end of a dimension with unknown size, it is recommended to pass
    in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

    Example 1:

    ```
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    steps = [1, 2]
    result = [
        [5, 7],
    ]
    ```

    Example 2:

    ```
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
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
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$starts,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$ends,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$steps
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
