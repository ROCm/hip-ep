//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SoftmaxCrossEntropyLossOp
//===----------------------------------------------------------------------===//

def Onnx_SoftmaxCrossEntropyLossOp : Onnx_Op<"SoftmaxCrossEntropyLoss", [Pure]> {
  let summary = "ONNX SoftmaxCrossEntropyLoss, schema version 13";

  let description = [{
    Loss function that measures the softmax cross entropy
    between 'scores' and 'labels'.
    This operator first computes a loss tensor whose shape is identical to the labels input.
    If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
    If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
    the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
    After L is available, this operator can optionally do a reduction operator.

    * shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
      with K >= 1 in case of K-dimensional loss.
    * shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
      with K >= 1 in case of K-dimensional loss.

    The loss for one sample, l_i, can calculated as follows:
    ```
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
    ```
    or
    ```
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
    ```

    loss is zero for the case when label-value equals ignore_index.
    ```
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
    ```

    where:
    ```
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]
    ```

    Finally, L is optionally reduced:

    * If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
    * If reduction = 'sum', the output is scalar: Sum(L).
    * If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
      where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$scores,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$labels,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      NoneType
    ]>:$weights,
    OptionalAttr<SI64Attr>:$ignore_index,
    DefaultValuedStrAttr<StrAttr, "mean">:$reduction
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output,
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>,
      NoneType
    ]>:$log_prob
  );
}
