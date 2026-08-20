//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RNNOp
//===----------------------------------------------------------------------===//

def Onnx_RNNOp : Onnx_Op<"RNN", [Pure]> {
  let summary = "ONNX RNN, schema version 22";

  let description = [{
    Computes an one-layer simple RNN. This operator is usually supported
    via some custom implementation such as CuDNN.

    Notations:

    * `X` - input tensor
    * `i` - input gate
    * `t` - time step (t-1 means previous time step)
    * `Wi` - W parameter weight matrix for input gate
    * `Ri` - R recurrence weight matrix for input gate
    * `Wbi` - W parameter bias vector for input gate
    * `Rbi` - R parameter bias vector for input gate
    * `WBi` - W parameter weight matrix for backward input gate
    * `RBi` - R recurrence weight matrix for backward input gate
    * `WBbi` - WR bias vectors for backward input gate
    * `RBbi` - RR bias vectors for backward input gate
    * `H` - Hidden state
    * `num_directions` - 2 if direction == bidirectional else 1

    Activation functions:

    * Relu(x)                - max(0, x)
    * Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
    * Sigmoid(x)             - 1/(1 + e^{-x})

    NOTE: Below are optional

    * Affine(x)              - alpha*x + beta
    * LeakyRelu(x)           - x if x >= 0 else alpha * x
    * ThresholdedRelu(x)     - x if x >= alpha else 0
    * ScaledTanh(x)          - alpha*Tanh(beta*x)
    * HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
    * Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
    * Softsign(x)            - x/(1 + |x|)
    * Softplus(x)            - log(1 + e^x)

    Equations (Default: f=Tanh):

    * Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$W,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$R,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$B,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$initial_h,
    OptionalAttr<F32ArrayAttr>:$activation_alpha,
    OptionalAttr<F32ArrayAttr>:$activation_beta,
    DefaultValuedAttr<StrArrayAttr, "{\"Tanh\", \"Tanh\"}">:$activations,
    OptionalAttr<F32Attr>:$clip,
    DefaultValuedStrAttr<StrAttr, "forward">:$direction,
    OptionalAttr<SI64Attr>:$hidden_size,
    DefaultValuedAttr<SI64Attr, "0">:$layout
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$Y,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$Y_h
  );
}
