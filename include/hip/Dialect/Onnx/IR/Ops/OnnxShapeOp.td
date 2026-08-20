//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ShapeOp
//===----------------------------------------------------------------------===//

def Onnx_ShapeOp : Onnx_Op<"Shape", [Pure]> {
  let summary = "ONNX Shape, schema version 25";

  let description = [{
    Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
    Optional attributes start and end can be used to compute a slice of the input tensor's shape.
    If start axis is omitted, the slice starts from axis 0.
    The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
    If the end axis is omitted, the axes upto the last one will be included.
    Negative axes indicate counting back from the last axis.
    Note that axes will be clamped to the range [0, r], where r is the
    rank of the input tensor if they are out-of-range (after adding r in the case of
    negative axis). Thus, specifying any end value > r is equivalent to specifying an end
    value of r, and specifying any start value < -r is equivalent to specifying a start
    value of 0. If start > end, the result will be an empty shape.

    Examples:

    ```
    Input tensor with shape: [2, 3, 4]
    No attributes specified.
    Output: [2, 3, 4]
    ```

    ```
    Input tensor with shape: [2, 3, 4]
    start: -1
    Output: [4]
    ```

    ```
    Input tensor with shape: [2, 3, 4]
    end: -1
    Output: [2, 3]
    ```

    ```
    Input tensor with shape: [2, 3, 4]
    start: 1
    end: 2
    Output: [3]
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>:$data,
    OptionalAttr<SI64Attr>:$end,
    DefaultValuedAttr<SI64Attr, "0">:$start
  );

  let results = (outs
    TensorOf<[I64]>:$shape
  );
}
