//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RMSNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_RMSNormalizationOp : Onnx_Op<"RMSNormalization", [Pure]> {
  let summary = "ONNX RMSNormalization, schema version 23";

  let description = [{
          This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.org/pdf/1910.07467.
          The overall computation can be split into two stages. The root mean squared norm is taken over the last D dimensions,
          where D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5) (a 2-dimensional shape),
          the rms norm is computed over the last 2 dimensions of the input. The computation required by standardization can be
          described by the following equations.
          ```
          XSquared = Mul(X, X)
          XSquaredMean = ReduceMean<axes=normalized_axes>(XSquared)
          MeanSquareEpsilon = Add(XSquaredMean, epsilon)
          RMS = Sqrt(MeanSquareEpsilon)
          Normalized = Div(X, RMS)
          ```
          where `normalized_axes` is `[axis, ..., rank of X - 1]`. The variables `RMS` stand for root mean square,
          Depending on `stash_type` attribute, the actual computation
          must happen in different floating-point precision.
          For example, if `stash_type` is 1, this operator casts
          all input variables to 32-bit float, perform the computation, and
          finally cast `Normalized` back to the original type of `X`.
          The second stage then scales the outcome of the first stage using:
          ```
          Y= Mul(Normalized, Scale)
          ```
          Let `d[i]` indicate the i-th dimension of `X`.
          If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
          the shape of `RMS` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
          `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
          (`Scale` should be unidirectional broadcastable to tensor `X`);
          for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$scale,
    DefaultValuedAttr<SI64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<SI64Attr, "1">:$stash_type
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
