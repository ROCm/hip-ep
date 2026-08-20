//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_SoftmaxOp
//===----------------------------------------------------------------------===//

def Onnx_SoftmaxOp : Onnx_Op<"Softmax", [Pure]> {
  let summary = "ONNX Softmax, schema version 13";

  let description = [{
    The operator computes the normalized exponential values for the given input:

     Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

    The "axis" attribute indicates the dimension along which Softmax
    will be performed. The output tensor has the same shape
    and contains the Softmax values of the corresponding input.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$input,
    DefaultValuedAttr<SI64Attr, "-1">:$axis
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output
  );
}
