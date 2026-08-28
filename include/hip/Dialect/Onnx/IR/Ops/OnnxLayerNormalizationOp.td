//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LayerNormalizationOp
//===----------------------------------------------------------------------===//

def Onnx_LayerNormalizationOp : Onnx_Op<"LayerNormalization", [Pure]> {
  let summary = "ONNX LayerNormalization, schema version 17";

  let description = [{
          This is layer normalization defined in ONNX as function.
          The overall computation can be split into two stages.
          The first stage is standardization, which makes the
          normalized elements have zero mean and unit variances.
          The computation required by standardization can be
          described by the following equations.
          ```
          Mean = ReduceMean<axes=normalized_axes>(X)
          D = Sub(X, Mean)
          DD = Mul(D, D)
          Var = ReduceMean<axes=normalized_axes>(DD)
          VarEps = Add(Var, epsilon)
          StdDev = Sqrt(VarEps)
          InvStdDev = Reciprocal(StdDev)
          Normalized = Mul(D, InvStdDev)
          ```
          where `normalized_axes` is `[axis, ..., rank of X - 1]`.
          The variables `Var` and `StdDev` stand for variance and
          standard deviation, respectively. The second output is
          `Mean` and the last one is `InvStdDev`.
          Depending on `stash_type` attribute, the actual computation
          must happen in different floating-point precision.
          For example, if `stash_type` is 1, this operator casts
          all input variables to 32-bit float, perform the computation, and
          finally cast `Normalized` back to the original type of `X`.
          The second stage then scales and shifts the outcome of the
          first stage using
          ```
          NormalizedScaled = Mul(Normalized, Scale)
          Y = Add(NormalizedScaled, B)
          ```
          The second stage doesn't depends on `stash_type`.
          All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
          The same variable (i.e., input, output, and attribute) uses
          the same name in the equations above and this operator's definition.
          Let `d[i]` indicate the i-th dimension of `X`.
          If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
          the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
          `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
          (tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
          for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Scale,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      NoneType
    ]>:$B,
    DefaultValuedAttr<SI64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
    DefaultValuedAttr<SI64Attr, "1">:$stash_type
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType]>:$Mean,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType]>:$InvStdDev
  );
}
