//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_UniqueOp
//===----------------------------------------------------------------------===//

def Onnx_UniqueOp : Onnx_Op<"Unique", [Pure]> {
  let summary = "ONNX Unique, schema version 11";

  let description = [{
    Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
    Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

    This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
    The first output tensor 'Y' contains all unique values or subtensors of the input.
    The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
    The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
    The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

    Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

    https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

    Example 1:
    ```
    input_X = [2, 1, 1, 3, 4, 3]
    attribute_sorted = 0
    attribute_axis = None
    output_Y = [2, 1, 3, 4]
    output_indices = [0, 1, 3, 4]
    output_inverse_indices = [0, 1, 1, 2, 3, 2]
    output_counts = [1, 2, 2, 1]
    ```

    Example 2:
    ```
    input_X = [[1, 3], [2, 3]]
    attribute_sorted = 1
    attribute_axis = None
    output_Y = [1, 2, 3]
    output_indices = [0, 2, 1]
    output_inverse_indices = [0, 2, 1, 2]
    output_counts = [1, 1, 2]
    ```

    Example 3:
    ```
    input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
    attribute_sorted = 1
    attribute_axis = 0
    output_Y = [[1, 0, 0], [2, 3, 4]]
    output_indices = [0, 2]
    output_inverse_indices = [0, 0, 1]
    output_counts = [2, 1]
    ```

    Example 4:
    ```
    input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
                [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
    attribute_sorted = 1
    attribute_axis = 1
    ```

    intermediate data are presented below for better understanding:
    there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
    ```
    A: [[1, 1], [1, 1]],
       [[0, 1], [0, 1]],
       [[2, 1], [2, 1]],
       [[0, 1], [0, 1]].
    ```

    there are 3 unique subtensors:
    ```
    [[1, 1], [1, 1]],
    [[0, 1], [0, 1]],
    [[2, 1], [2, 1]].
    ```

    sorted unique subtensors:
    ```
    B: [[0, 1], [0, 1]],
       [[1, 1], [1, 1]],
       [[2, 1], [2, 1]].
    ```

    output_Y is constructed from B:
    ```
    [[[0. 1.], [1. 1.], [2. 1.]],
     [[0. 1.], [1. 1.], [2. 1.]]]
    ```

    output_indices is to map from B to A:
    ```
    [1, 0, 2]
    ```

    output_inverse_indices is to map from A to B:
    ```
    [1, 0, 2, 0]
    ```

    output_counts:
    ```
    [2, 1, 1]
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$X,
    OptionalAttr<SI64Attr>:$axis,
    DefaultValuedAttr<SI64Attr, "1">:$sorted
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>,
      TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$Y,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$indices,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$inverse_indices,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$counts
  );
}
