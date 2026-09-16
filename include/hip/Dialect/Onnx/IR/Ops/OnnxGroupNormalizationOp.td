//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GroupNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_GroupNormalizationOp : Onnx_Op<"GroupNormalization", [Pure]> {
  let summary = "ONNX GroupNormalization, schema version 21";

  let description = [{
    A GroupNormalization function. Carries out group normalization as described in
    the paper https://arxiv.org/abs/1803.08494

    This operator transforms input according to
    ```
    y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
    ```
    where the mean and variance are computed per instance per group of channels, and
    `scale` and `bias` should be specified for each channel. The number of
    groups `num_groups` should be divisible by the number of channels so that there are
    an equal number of channels per group.

    The overall computation has two stages: the first stage normalizes the elements to
    have zero mean and unit variance for each instance in each group, and the second
    stage scales and shifts the results of the first stage. The floating-point precision
    used in the first stage is determined by the `stash_type` attribute. For example,
    if `stash_type` is 1, the operator casts all input variables to 32-bit float,
    performs the computation, and finally casts the normalized results back to the
    original type of `X`. The second stage does not depend on `stash_type`.

    When the number of groups is the same as the number of channels, this operator is
    equivalent to InstanceNormalization. When there is only one group, this operator
    is equivalent to LayerNormalization.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$scale,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$bias,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
    SI64Attr:$num_groups,
    DefaultValuedAttr<SI64Attr, "1">:$stash_type
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
