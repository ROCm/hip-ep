//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_InstanceNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_InstanceNormalizationOp : Onnx_Op<"InstanceNormalization", [Pure]> {
  let summary = "ONNX InstanceNormalization, schema version 22";

  let description = [{
    Carries out instance normalization as described in the paper
    https://arxiv.org/abs/1607.08022.

    y = scale * (x - mean) / sqrt(variance + epsilon) + B,
    where mean and variance are computed per instance per channel.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$input,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$scale,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$B,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
