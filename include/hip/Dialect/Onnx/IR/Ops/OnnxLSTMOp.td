//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LSTMOp
//===----------------------------------------------------------------------===//

def Onnx_LSTMOp : Onnx_Op<"LSTM", [Pure]> {
  let summary = "ONNX LSTM, schema version 22";

  let description = [{
    Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    Notations:

    * `X` - input tensor
    * `i` - input gate
    * `o` - output gate
    * `f` - forget gate
    * `c` - cell gate
    * `t` - time step (t-1 means previous time step)
    * `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
    * `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
    * `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
    * `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
    * `P[iof]`  - P peephole weight vector for input, output, and forget gates
    * `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
    * `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
    * `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
    * `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
    * `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
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

    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    * it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    * ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    * ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    * Ct = ft (.) Ct-1 + it (.) ct
    * ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    * Ht = ot (.) h(Ct)
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
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$initial_c,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$P,
    OptionalAttr<F32ArrayAttr>:$activation_alpha,
    OptionalAttr<F32ArrayAttr>:$activation_beta,
    OptionalAttr<StrArrayAttr>:$activations,
    OptionalAttr<F32Attr>:$clip,
    DefaultValuedStrAttr<StrAttr, "forward">:$direction,
    OptionalAttr<SI64Attr>:$hidden_size,
    DefaultValuedAttr<SI64Attr, "0">:$input_forget,
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
    ]>:$Y_h,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$Y_c
  );
}
