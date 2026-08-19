//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AffineGridOp
//===----------------------------------------------------------------------===//

def Onnx_AffineGridOp : Onnx_Op<"AffineGrid", [Pure]> {
  let summary = "ONNX AffineGrid, schema version 20";

  let description = [{
    Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
    (https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
    An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
    ```
    [r00, r01, r02, t0]   [x]   [x']
    [r10, r11, r12, t1] * [y] = [y']
    [r20, r21, r22, t2]   [z]   [z']
    [0,   0,   0,   1 ]   [1]   [1 ]
    ```
    where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
    The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

    Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
    The output `grid` contains positions in the output space.

    When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
    ```
    v            v            v            v
    |-------------------|------------------|
    -1                  0                  1
    ```
    When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
    ```
        v        v         v         v
    |------------------|-------------------|
    -1                 0                   1
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$theta,
    TensorOf<[I64]>:$size,
    DefaultValuedAttr<SI64Attr, "0">:$align_corners
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$grid
  );
}
