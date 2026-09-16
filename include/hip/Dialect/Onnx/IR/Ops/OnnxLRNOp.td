//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LRNOp
//===----------------------------------------------------------------------===//

def Onnx_LRNOp : Onnx_Op<"LRN", [Pure]> {
  let summary = "ONNX LRN, schema version 13";

  let description = [{
    Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
    It normalizes over local input regions.
    The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
    of shape `(N x C x D1 x D2, ..., Dk)`, its region is
    `{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

    `square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
    where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

    `Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X,
    DefaultValuedAttr<F32Attr, "0.0001">:$alpha,
    DefaultValuedAttr<F32Attr, "0.75">:$beta,
    DefaultValuedAttr<F32Attr, "1.0">:$bias,
    SI64Attr:$size
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
