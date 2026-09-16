//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_PadOp
//===----------------------------------------------------------------------===//

def Onnx_PadOp : Onnx_Op<"Pad", [Pure]> {
  let summary = "ONNX Pad, schema version 25";

  let description = [{
    Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
    a padded tensor (`output`) is generated.

    The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

    1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

    2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

    3) `edge` - pads with the edge values of array

    4) `wrap` - wrap-around padding as if the data tensor forms a torus


    Example 1 (`constant` mode):

    Insert 0 pads to the beginning of the second dimension.

    ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'constant'

    constant_value = 0.0

    output = [
        [0.0, 0.0, 1.0, 1.2],
        [0.0, 0.0, 2.3, 3.4],
        [0.0, 0.0, 4.5, 5.7],
    ]
    ```

    Example 2 (`reflect` mode):

    ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'reflect'

    output = [
        [1.0, 1.2, 1.0, 1.2],
        [2.3, 3.4, 2.3, 3.4],
        [4.5, 5.7, 4.5, 5.7],
    ]
    ```

    Example 3 (`edge` mode):

    ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'edge'

    output = [
        [1.0, 1.0, 1.0, 1.2],
        [2.3, 2.3, 2.3, 3.4],
        [4.5, 4.5, 4.5, 5.7],
    ]
    ```

    Example 4 (`wrap` mode):

    ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [2, 1, 1, 1]

    mode = 'wrap'

    output = [
        [3.4, 2.3, 3.4, 2.3],
        [5.7, 4.5, 5.7, 4.5],
        [1.2, 1.0, 1.2, 1.0],
        [3.4, 2.3, 3.4, 2.3],
        [5.7, 4.5, 5.7, 4.5],
        [1.2, 1.0, 1.2, 1.0],
    ]
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
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$pads,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>, NoneType
    ]>:$constant_value,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
    DefaultValuedStrAttr<StrAttr, "constant">:$mode
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>:$output
  );
}
