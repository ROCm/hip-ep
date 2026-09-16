//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_HardmaxOp
//===----------------------------------------------------------------------===//

def Onnx_HardmaxOp : Onnx_Op<"Hardmax", [Pure]> {
  let summary = "ONNX Hardmax, schema version 13";

  let description = [{
    The operator computes the hardmax values for the given input:

     Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

    The "axis" attribute indicates the dimension along which Hardmax
    will be performed. The output tensor has the same shape
    and contains the Hardmax values of the corresponding input.
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
