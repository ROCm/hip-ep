//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LogSoftmaxOp
//===----------------------------------------------------------------------===//

def Onnx_LogSoftmaxOp : Onnx_Op<"LogSoftmax", [Pure]> {
  let summary = "ONNX LogSoftmax, schema version 13";

  let description = [{
    The operator computes the log of softmax values for the given input:

     LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

    The "axis" attribute indicates the dimension along which LogSoftmax
    will be performed. The output tensor has the same shape
    and contains the LogSoftmax values of the corresponding input.
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
